"""
popoe.adapters — the reference stage implementations. Thin wrappers that make the
concrete FreeZe-style classes (feature extractors, RANSAC/ICP/scoring functions)
satisfy the stage Protocols in popoe.interfaces, without changing their logic, so
they compose in `interfaces.Pipeline`. `examples/pipeline_selfcheck.py` checks the
chain is bitwise-identical to the inline `FreeZeV2.estimate_pose` body.

Two design points worth knowing:

  * The target encoder needs the query side's fitted visual PCA. Because fusion
    is an injectable component (popoe.fusion), we SHARE one fusion instance
    across both encoders — PCA reuse is automatic, no `_pca_vis` copy.
    `make_freeze_encoders()` wires that up.
  * `ICPRefiner` moves geometry only; the final feature scoring lives in the
    separate `FreeZeScorer` stage (see interfaces.PoseScorer). `refine` still
    takes `query` because ICP aligns the query point cloud (geometry), not for
    scoring.

Encoder adapters need the heavy models (DINOv2/GeDi) and a GPU; the solver /
refiner / selector adapters are pure numpy+open3d and unit-testable offline.
"""

from __future__ import annotations
import numpy as np

from popoe.interfaces import (
    Scene, ObjectModel, Detection, CanonFrame, PointFeatures, PoseHypothesis,
)


# ── Segmentation ────────────────────────────────────────────────────────

class PrecomputedSegmentor:
    """Wrap an already-computed list of (mask, score) as a Segmentor. Covers the
    GT-mask and public-CNOS-detection modes, where masks come from disk rather
    than being generated here. `provider(scene, obj) -> list[Detection]`."""

    def __init__(self, provider):
        self._provider = provider

    def segment(self, scene: Scene, obj: ObjectModel) -> list[Detection]:
        return self._provider(scene, obj)


# ── Feature encoders ────────────────────────────────────────────────────

def _intrinsics_dict(K: np.ndarray) -> dict:
    return {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2])}


class FreeZeQueryEncoder:
    """Adapt QueryFeatureExtractor. Produces PointFeatures whose meta carries the
    CanonFrame (derived from the sampled points, per the live convention) so the
    target side and solver can reuse it."""

    def __init__(self, extractor, n_points: int = 3000):
        self.ex = extractor
        self.n_points = n_points

    def encode_query(self, obj: ObjectModel) -> PointFeatures:
        import trimesh, torch
        # Reset PCA per object so each fits its own (matches eval scripts).
        self.ex._pca_vis = None
        mesh = trimesh.load(obj.mesh_path, force="mesh")
        pts, _ = trimesh.sample.sample_surface_even(mesh, self.n_points)
        pts = (pts / 1000.0).astype(np.float32)          # BOP mm -> m
        feats, pts_q = self.ex.extract_query_features(obj.mesh_path, torch.from_numpy(pts))
        pts_q = pts_q.numpy() if hasattr(pts_q, "numpy") else np.asarray(pts_q)
        # The fitted visual PCA is PER OBJECT. It is snapshotted here because the
        # fusion instance is SHARED with the target encoder: encoding another
        # object's query overwrites fusion.pca_vis, so any caller that
        # interleaves objects (e.g. an image-major eval loop) must re-install
        # this snapshot via FreeZeTargetEncoder.install_pca() before encoding
        # targets. (Measured failure: texture-reliant objects crater, geometry-
        # strong ones survive — a quiet cross-object feature corruption.)
        return PointFeatures(
            pts=pts_q, feats=feats,
            meta={"canon_frame": self.ex.canon_frame, "fusion": self.ex.fusion,
                  "pca_vis": self.ex.fusion.pca_vis},
        )


class FreeZeTargetEncoder:
    """Adapt TargetFeatureExtractor. Consumes the CanonFrame produced by the
    query side; relies on the shared fusion (see make_freeze_encoders) for the
    reused PCA, so no `_pca_vis` copy is needed here."""

    def __init__(self, extractor):
        self.ex = extractor

    def install_pca(self, pca_vis) -> None:
        """Install a query's visual-PCA snapshot (PointFeatures.meta['pca_vis'])
        before encoding its targets. Required whenever queries for multiple
        objects are encoded before their targets — see FreeZeQueryEncoder."""
        self.ex.fusion.pca_vis = pca_vis

    def encode_target(self, scene: Scene, det: Detection,
                      obj: ObjectModel, frame: CanonFrame) -> PointFeatures:
        self.ex._canon_scale = frame.scale          # convention from query side
        pts, feats = self.ex.extract_target_features(
            scene.rgb, scene.depth, det.mask, _intrinsics_dict(scene.K),
        )
        if pts is None:
            return PointFeatures(pts=np.empty((0, 3), np.float32),
                                 feats=np.empty((0, 1), np.float32))
        return PointFeatures(pts=np.asarray(pts), feats=np.asarray(feats))


def make_freeze_encoders(query_extractor, target_extractor, n_points: int = 3000):
    """Wire query+target extractors to SHARE one fusion instance (so the visual
    PCA fit on the query side is transparently reused on the target side), and
    return (QueryEncoder, TargetEncoder) adapters."""
    target_extractor.fusion = query_extractor.fusion
    return (FreeZeQueryEncoder(query_extractor, n_points),
            FreeZeTargetEncoder(target_extractor))


# ── Pose solve / refine / select ────────────────────────────────────────

class RansacSolver:
    """Adapt ransac_pose_estimation -> one coarse PoseHypothesis (s_coarse)."""

    def __init__(self, n_ransac: int = 10000, tau_inlier: float = 0.03, k: int = 10):
        self.n_ransac = n_ransac
        self.tau_inlier = tau_inlier
        self.k = k

    def solve(self, query: PointFeatures, target: PointFeatures,
              frame: CanonFrame) -> list[PoseHypothesis]:
        from popoe.pose_estimator import ransac_pose_estimation
        if len(target.pts) < 4:
            return []
        R, t, s = ransac_pose_estimation(
            query.pts, query.feats, target.pts, target.feats,
            n_iters=self.n_ransac, tau_inlier=self.tau_inlier, k=self.k,
        )
        return [PoseHypothesis(R=R, t=t, score=s, breakdown={"s_coarse": s})]


class ICPRefiner:
    """Adapt icp_refinement — GEOMETRY ONLY (coupling point #3). ICP aligns the
    query cloud to the dense target and records its fitness as s_icp; it does NOT
    compute the feature score (that is FreeZeScorer's job). The provisional score
    (s_coarse) is carried through untouched for FreeZeScorer to finalise."""

    def __init__(self, tau_icp: float = 0.03):
        self.tau_icp = tau_icp

    def refine(self, pose: PoseHypothesis, scene: Scene, obj: ObjectModel,
               query: PointFeatures, target: PointFeatures) -> PoseHypothesis:
        from popoe.pose_estimator import icp_refinement
        dense = target.pts_dense if target.pts_dense is not None else target.pts
        R_f, t_f, s_icp = icp_refinement(query.pts, dense, pose.R, pose.t, self.tau_icp)
        return PoseHypothesis(
            R=R_f, t=t_f, score=pose.score,     # provisional; FreeZeScorer sets final
            breakdown={**pose.breakdown, "s_icp": s_icp, "fitness": s_icp},
        )


class FreeZeScorer:
    """Adapt feature_aware_score + final_score into the single scoring stage.
    Reproduces FreeZeV2.estimate_pose's final combination exactly:
    s_fine re-scored at the refined pose, then S = s_coarse^a * s_fine^b * s_icp^g."""

    def __init__(self, tau_inlier: float = 0.03,
                 alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        self.tau_inlier = tau_inlier
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def score(self, pose: PoseHypothesis,
              query: PointFeatures, target: PointFeatures) -> PoseHypothesis:
        from popoe.pose_estimator import feature_aware_score, final_score
        s_fine, _ = feature_aware_score(
            pose.R, pose.t, query.pts, target.pts, query.feats, target.feats, self.tau_inlier,
        )
        s_coarse = pose.breakdown.get("s_coarse", pose.score)
        s_icp = pose.breakdown.get("s_icp", pose.breakdown.get("fitness", 1.0))
        score = final_score(s_coarse, s_fine, s_icp, self.alpha, self.beta, self.gamma)
        return PoseHypothesis(
            R=pose.R, t=pose.t, score=score,
            breakdown={**pose.breakdown, "s_fine": s_fine},
        )


class BestScoreSelector:
    """Pick the highest-scoring hypothesis (the multi-mask top-K choice)."""

    def select(self, candidates: list[PoseHypothesis]):
        cands = [c for c in candidates if c is not None]
        return max(cands, key=lambda h: h.score) if cands else None
