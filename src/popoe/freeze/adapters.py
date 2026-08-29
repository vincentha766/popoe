"""
popoe.freeze.adapters — the FreeZe-v2 stage implementations. Thin wrappers that
make the concrete FreeZe classes (feature extractors, feature-aware scoring)
satisfy the stage Protocols in popoe.interfaces, without changing their logic.
`examples/pipeline_selfcheck.py` checks the full adapter chain is
bitwise-identical to the inline `FreeZeV2.estimate_pose` body
(`examples/freezev2_monolith.py`).

The method-agnostic adapters (RansacSolver, ICPRefiner, BestScoreSelector)
stay in popoe.adapters.

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


def sample_query_surface(mesh_path: str, n_points: int, seed: int) -> np.ndarray:
    """P_Q^raw — the query surface sample, in the mesh's own units (mm for BOP).

    THE single live sampling site for the query cloud: `extract_query_features`
    takes the cloud from its caller, so this is where P_Q^raw is decided.

    `POPOE_QUERY_SAMPLER` selects the sampler and IS a cache key
    (popoe.cache.CONDITIONAL_ENC_KEYS), so switching it invalidates query
    features instead of silently reusing them.

    - ``even`` (default; every published number ran under it):
      ``trimesh.sample.sample_surface_even`` — rejection-based, approximately
      even. Measured 2026-08-10 on six YCB-V meshes at N=5000: returns the full
      count, but minimum nearest-neighbour spacing is 22% below the poisson arm.
    - ``poisson``: Open3D's Yuksel-2015 sample elimination (blue noise, minimum
      inter-point distance). FreeZeV2 Sec. III-D cites Bridson 2007 [77], which
      is a VOLUMETRIC dart-throwing algorithm — there is no unique way to *be*
      it on a mesh surface, so this satisfies the property the paper states
      rather than being the cited algorithm. Isolation arm D20 only; it is NOT
      the mainline recipe and must not be described as "implementing [77]".
    """
    import os
    sampler = os.environ.get("POPOE_QUERY_SAMPLER", "even")
    if sampler == "even":
        import trimesh
        mesh = trimesh.load(mesh_path, force="mesh")
        pts, _ = trimesh.sample.sample_surface_even(mesh, n_points, seed=seed)
        return np.asarray(pts)
    if sampler == "poisson":
        import open3d as o3d
        # Open3D's poisson sampler draws from a PROCESS-GLOBAL RNG: two calls
        # without reseeding return different clouds (measured). trimesh takes a
        # per-call seed; copying that shape here would leave a silently
        # non-reproducible arm, so reseed immediately before each sample.
        o3d.utility.random.seed(int(seed))
        om = o3d.io.read_triangle_mesh(mesh_path)
        pcd = om.sample_points_poisson_disk(number_of_points=n_points,
                                            init_factor=5)
        return np.asarray(pcd.points)
    # Deliberately fatal: falling back to the default on a typo is exactly how a
    # run ends up not being the run you think it is.
    raise ValueError(
        f"POPOE_QUERY_SAMPLER must be 'even' or 'poisson', got {sampler!r}")


def query_sampler_provenance(n_points: int) -> str:
    """One log line naming the sampler actually in force. An isolation arm whose
    identity is not printed is not isolable (see recipes.solver_provenance and
    the C9/C9b `--corr-topk` incident, 2026-08-08/09)."""
    import os
    sampler = os.environ.get("POPOE_QUERY_SAMPLER", "even")
    return (f"query_sampler={sampler} n_points={n_points} "
            f"(seed=per-object obj_id unless --seed overrides)")


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
        import torch
        # Reset PCA per object so each fits its own (matches eval scripts).
        self.ex._pca_vis = None
        seed = self.seed if self.seed is not None else obj.obj_id
        pts = sample_query_surface(obj.mesh_path, self.n_points, seed)
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
        objects are encoded before their targets — see FreeZeQueryEncoder.

        `None` is REFUSED, loudly. It reads like "no snapshot to install", but
        DinoGeDiFusion.fuse treats `pca_vis is None` as "fit one now" — and the
        target side has enough valid points to succeed, so the targets end up
        projected in a basis fitted from TARGET data while the cached query
        features live in the query basis. Cosines are then compared across two
        unrelated bases: no exception, no warning, just scrambled similarity.
        That is the PCA-basis incoherence ISSUES.md root-caused (measured AR
        0.16-0.25 vs 0.79-0.85 on YCB-V obj8), arriving through a second door.

        The live caller is an incomplete cache entry — query arrays present, PCA
        sidecar missing (examples/bop_eval.py). A degenerate query whose PCA
        genuinely never fitted lands here too, and must also raise: its features
        were built by truncation, so a target that fits a real PCA is just as
        incoherent. Either way the answer is re-encode, never substitute.

        A query that legitimately needed NO reduction (`n_vis == vis_dim`, e.g.
        `POPOE_VIS_DIM=1536` against 1536-D DINOv2) is a different thing and is
        accepted: it hands over `fusion.IdentityReduction()`, not `None`. That
        distinction is the whole reason the marker exists — `None` has to stay
        reserved for "missing", or a lost sidecar becomes indistinguishable
        from a deliberate configuration."""
        if pca_vis is None:
            raise ValueError(
                "install_pca(None): the target side must REUSE the query's "
                "visual reduction, never fit its own. A None snapshot means the "
                "query features are unusable (incomplete cache entry, or a query "
                "whose PCA never fitted) — re-encode the query instead of "
                "installing nothing. A query that genuinely needed no reduction "
                "passes fusion.IdentityReduction(), not None. See the "
                "availability contract in popoe.interfaces.")
        self.ex.fusion.pca_vis = pca_vis

    def encode_target(self, scene: Scene, det: Detection,
                      obj: ObjectModel, frame: CanonFrame) -> PointFeatures:
        from popoe import profiling
        self.ex._canon_scale = frame.scale          # convention from query side
        with profiling.stage("target_encode"):
            pts, feats = self.ex.extract_target_features(
                scene.rgb, scene.depth, det.mask, _intrinsics_dict(scene.K),
            )
        if pts is None:
            reason = getattr(self.ex, "_last_target_skip_reason", None)
            return PointFeatures(pts=np.empty((0, 3), np.float32),
                                 feats=np.empty((0, 1), np.float32),
                                 meta={"detection": det,
                                       "skip_reason": reason or "not encodable"})
        # Keep the Detection on meta so post-ICP stages (e.g. render re-rank)
        # can recover the mask/bbox without re-segmenting.
        return PointFeatures(pts=np.asarray(pts), feats=np.asarray(feats),
                             meta={"detection": det})


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
