"""
popoe.freeze.adapters — the FreeZe-v2 stage implementations. Thin wrappers that
make the concrete FreeZe classes (feature extractors, feature-aware scoring)
satisfy the stage Protocols in popoe.interfaces, without changing their logic.
`examples/pipeline_selfcheck.py` checks the full adapter chain is
bitwise-identical to the inline `FreeZeV2.estimate_pose` body
(`examples/freezev2_monolith.py`).

The method-agnostic adapters (RansacSolver, ICPRefiner, BestScoreSelector,
PrecomputedSegmentor, ...) stay in popoe.adapters.

One design point worth knowing: the target encoder needs the query side's
fitted visual PCA. Because fusion is an injectable component
(popoe.freeze.fusion), we SHARE one fusion instance across both encoders —
PCA reuse is automatic, no `_pca_vis` copy. `make_freeze_encoders()` wires
that up.

Encoder adapters need the heavy models (DINOv2/GeDi) and a GPU; FreeZeScorer
is pure numpy and unit-testable offline.
"""

from __future__ import annotations
import numpy as np

from popoe.interfaces import (
    Scene, ObjectModel, Detection, CanonFrame, PointFeatures, PoseHypothesis,
)


# ── Feature encoders ────────────────────────────────────────────────────

def _intrinsics_dict(K: np.ndarray) -> dict:
    return {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2])}


class FreeZeQueryEncoder:
    """Adapt QueryFeatureExtractor. Produces PointFeatures whose meta carries the
    CanonFrame (derived from the sampled points, per the live convention) so the
    target side and solver can reuse it."""

    def __init__(self, extractor, n_points: int = 3000, seed: int | None = None):
        self.ex = extractor
        self.n_points = n_points
        # Deterministic surface sampling by default (seed = obj_id): unseeded
        # sampling makes query features differ per RUN, which compounds with
        # solver stochasticity into run-to-run AR variance (see ISSUES.md).
        self.seed = seed

    @property
    def render_backend(self) -> str:
        """Which renderer produces the CAD views these features come from —
        'nvdiffrast' or 'trimesh'. Belongs in the cache key: the two are not
        interchangeable (see feature_extractor.QueryFeatureExtractor)."""
        return self.ex.render_backend

    def encode_query(self, obj: ObjectModel) -> PointFeatures:
        import trimesh, torch
        # Reset PCA per object so each fits its own (matches eval scripts).
        self.ex._pca_vis = None
        mesh = trimesh.load(obj.mesh_path, force="mesh")
        pts, _ = trimesh.sample.sample_surface_even(
            mesh, self.n_points,
            seed=self.seed if self.seed is not None else obj.obj_id)
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


# ── Scoring ─────────────────────────────────────────────────────────────

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
        from popoe.registration import feature_aware_score, final_score
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
