"""
popoe.interfaces — the stage contracts.

Data objects that flow between stages, and the Protocol each swappable stage
must satisfy. Implementations live in popoe.adapters, popoe.freeze,
popoe.registration, popoe.solvers. `Pipeline` is the reference composition;
the evaluated BOP loop is examples/bop_eval.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable
import numpy as np

from popoe import profiling


# ════════════════════════════════════════════════════════════════════════
# 0. The availability contract.
# ════════════════════════════════════════════════════════════════════════

class BackendUnavailable(RuntimeError):
    """A stage's backend is missing: no package, no checkpoint, no device.

    An implementation raises this INSTEAD of quietly substituting a weaker
    method. Two different methods behind one name is the bug this exists to
    prevent — it makes the reported result unattributable (which segmentor
    produced this mask? which renderer produced these templates?) and it
    poisons the config-addressed cache, whose key fingerprints the config you
    ASKED for, not the method that silently ran instead (see cache.py).

    Substitution is a CALLER's policy, not a silent fallback inside the stage.

    This is an *availability* signal, not an error channel. A runtime failure —
    CUDA OOM, a corrupt mesh — must propagate: "the fallback handled it" is how
    real bugs get buried."""


def is_runtime_failure(exc: BaseException) -> bool:
    """True for failures a backend-load guard must RE-RAISE, not convert.

    The guards around model loading are broad (`except Exception`) because the
    ways a checkpoint can be unreachable are many and boring. That breadth also
    catches out-of-memory, and an OOM reported as `BackendUnavailable` is the
    worst case the availability contract exists to prevent: a fallback chain
    routes around it, a weaker method answers, and the run looks fine.

    Matching the exception CLASS alone is not enough. Only torch >= 1.13 raises
    the dedicated `torch.cuda.OutOfMemoryError`; plenty of allocation failures
    still arrive as a bare `RuntimeError('CUDA out of memory. Tried to allocate
    ...')`. Matched by name and message so this module stays torch-free.

    The message policy allowlists ordinary CUDA availability cases, then treats
    CUDA/driver/library runtime-shaped errors as faults. The availability set is
    intentionally small and stable:

        no kernel image is available   arch mismatch (sm_89 kernels elsewhere)
        no CUDA-capable device         no GPU on this box
        invalid device ordinal         wrong device index
        driver version is insufficient driver too old

    Hard CUDA faults are more open-ended: a missed sticky fault can poison the
    context for the NEXT backend too, which then reports itself unavailable for
    the same reason. The chain runs to its end and the caller is told "no
    backend is available" while the real cause -- a failing card, a watchdog
    timeout, or a destroyed context -- never surfaces.

    Underscores are normalised so driver-style spellings
    (`CUDA_ERROR_OUT_OF_MEMORY`, `CUBLAS_STATUS_ALLOC_FAILED`) match too.
    """
    if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
        return True
    if not isinstance(exc, RuntimeError):
        return False
    text = str(exc).lower().replace("_", " ")

    availability = (
        "no kernel image is available",
        "no cuda-capable device",
        "no cuda capable device",
        "invalid device ordinal",
        "driver version is insufficient",
    )
    if any(reason in text for reason in availability):
        return False

    named_faults = (
        "out of memory", "alloc failed", "illegal memory access",
        "device-side assert", "device side assert",
        "unspecified launch failure", "misaligned address",
        "illegal instruction", "launch timed out",
        "ecc error", "uncorrectable")
    if any(fault in text for fault in named_faults):
        return True

    runtime_markers = (
        "cuda error", "cuda runtime error", "cuda driver error",
        "cudnn error", "cudnn status", "cublas error", "cublas status",
    )
    return any(marker in text for marker in runtime_markers)


# ════════════════════════════════════════════════════════════════════════
# 1. Cross-cutting data — constructed once, threaded through every stage.
#    These carry the conventions (units, canonicalisation) that were
#    previously implicit and re-derived per module (the #2 coupling point).
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Scene:
    """One RGB-D observation. depth is in METRES (already x depth_scale/1000)."""
    rgb: np.ndarray                     # (H, W, 3) uint8
    depth: np.ndarray                   # (H, W) float32, metres
    K: np.ndarray                       # (3, 3) camera intrinsics
    scene_id: int = -1
    im_id: int = -1


@dataclass(frozen=True)
class FrameManifest:
    """File-level description of one RGB-D frame.

    This is an I/O boundary, not a segmentation result: `detections_path`
    points to 2D masks/scores, while `depth_path` carries the depth image.
    Loaders must convert raw depth to metres before constructing `Scene`.
    """
    rgb_path: str
    depth_path: str
    K: np.ndarray
    depth_scale: float = 1.0             # metres per raw depth unit
    scene_id: int = -1
    im_id: int = -1
    detections_path: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        K = np.asarray(self.K, dtype=np.float64)
        if K.shape != (3, 3):
            if K.size != 9:
                raise ValueError(f"K must be 3x3 or flat length-9, got {K.shape}")
            K = K.reshape(3, 3)
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "depth_scale", float(self.depth_scale))
        object.__setattr__(self, "scene_id", int(self.scene_id))
        object.__setattr__(self, "im_id", int(self.im_id))


@dataclass(frozen=True)
class ObjectModel:
    """A target object: CAD mesh + BOP metadata. The single source of truth for
    diameter and symmetry, so downstream stages never re-guess them."""
    obj_id: int
    mesh_path: str                      # BOP .ply, vertices in mm
    diameter: float                     # metres — drives CanonFrame.scale
    symmetries: Sequence[np.ndarray] = field(default_factory=list)  # 4x4 transforms


@dataclass(frozen=True)
class CanonFrame:
    """Query/target registration scale: pts_canon = pts * scale.

    scale = 1 / max_extent of the QUERY sampled cloud (metres), NOT the BOP
    diameter. GeDi's r_lrf=0.5 m was trained on ~1 m scenes. Produced by query
    encoding and reused on the target side.
    """
    scale: float

    @classmethod
    def from_points(cls, pts: np.ndarray) -> "CanonFrame":
        extent_m = float(np.ptp(pts, axis=0).max())
        return cls(scale=1.0 / max(extent_m, 1e-6))


# ════════════════════════════════════════════════════════════════════════
# 2. Data that flows between stages.
# ════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """Output of segmentation: one candidate region for one object.

    `score` is only comparable WITHIN one segmentor: it is a DINO cosine
    similarity for the CNOS segmentors, SAM's predicted IoU for SAMSegmentor,
    and a mask AREA FRACTION for DepthSegmentor. Never merge-and-sort
    detections from different segmentors — `source` says which one produced
    this, and is what a fallback chain records (see segmentor.py)."""
    mask: np.ndarray                    # (H, W) bool
    score: float
    bbox: Optional[tuple] = None        # (x0, y0, x1, y1)
    descriptor: Optional[np.ndarray] = None  # e.g. CNOS CLS feature; may be None
    source: str = ""                    # segmentor that produced it, e.g. "cnos"


@dataclass
class PointFeatures:
    """Query AND target features share ONE schema so matching is symmetric.
    `feats` is already fused and L2-normed."""
    pts: np.ndarray                     # (N, 3) points, metres, camera or model frame
    feats: np.ndarray                   # (N, D) fused per-point descriptors
    pts_dense: Optional[np.ndarray] = None   # (M, 3) dense cloud for ICP
    meta: dict = field(default_factory=dict) # pca handle, canon scale echo, etc.


@dataclass
class PoseHypothesis:
    """A 6-DoF pose candidate with a score breakdown so selectors and ablations
    can inspect *why* it scored as it did (s_coarse / s_fine / s_icp / fitness)."""
    R: np.ndarray                       # (3, 3)
    t: np.ndarray                       # (3,) metres
    score: float
    breakdown: dict = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════
# 3. Swappable stages. Implement any one to extend the pipeline.
# ════════════════════════════════════════════════════════════════════════

@runtime_checkable
class Segmentor(Protocol):
    """Stage 0. File detections / CNOS / SAM / MUSE all fit here."""
    def segment(self, scene: Scene, obj: ObjectModel) -> list[Detection]: ...


@runtime_checkable
class PointDescriptor(Protocol):
    """Per-point 3D geometric descriptor (GeDi / FPFH).

    pts, pcd already scaled by CanonFrame (object extent ~= 1.0). Returns
    (N, D) float32; undescribed rows are NaN. `role` is "query" (CAD, all
    around) or "target" (single-view depth); role-blind backbones ignore it.
    """
    def compute(self, pts, pcd, role=None) -> np.ndarray: ...


@runtime_checkable
class FeatureFusion(Protocol):
    """Combine per-point visual and geometric features into one descriptor."""
    def fuse(self, vis_feats: np.ndarray, geo_feats: np.ndarray,
             apply_skip_vis: bool = False) -> np.ndarray: ...


@runtime_checkable
class QueryEncoder(Protocol):
    """Stage 1a. CAD -> sparse fused features. CanonFrame is an OUTPUT in
    PointFeatures.meta['canon_frame']."""
    def encode_query(self, obj: ObjectModel) -> PointFeatures: ...


@runtime_checkable
class TargetEncoder(Protocol):
    """Stage 1b. Masked RGB-D -> sparse fused features."""
    def encode_target(self, scene: Scene, det: Detection,
                      obj: ObjectModel, frame: CanonFrame) -> PointFeatures: ...


@runtime_checkable
class PoseSolver(Protocol):
    """Stage 2. Feature matching -> coarse pose(s). `frame` is unused by
    current solvers (scale is already applied in the features); kept so a
    solver that works in canonical space can read it."""
    def solve(self, query: PointFeatures, target: PointFeatures,
              frame: CanonFrame | None = None) -> list[PoseHypothesis]: ...


@runtime_checkable
class CoarseEstimator(Protocol):
    """External/direct coarse pose producer (SAM-6D PEM, MegaPose, …)."""
    def estimate(self, scene: Scene, obj: ObjectModel,
                 det: Optional[Detection] = None) -> list[PoseHypothesis]: ...


@runtime_checkable
class PoseRefiner(Protocol):
    """Stage 3. Move geometry (and report fitness). Scoring is PoseScorer."""
    def refine(self, pose: PoseHypothesis, scene: Scene, obj: ObjectModel,
               query: PointFeatures, target: PointFeatures) -> PoseHypothesis: ...


@runtime_checkable
class PoseScorer(Protocol):
    """Stage 3b. Final feature score on a (refined) hypothesis."""
    def score(self, pose: PoseHypothesis,
              query: PointFeatures, target: PointFeatures) -> PoseHypothesis: ...


@runtime_checkable
class Selector(Protocol):
    """Pick the winning hypothesis across candidate masks / poses."""
    def select(self, candidates: list[PoseHypothesis]) -> Optional[PoseHypothesis]: ...


@dataclass
class Pipeline:
    """Reference composition of the stage contracts.

    The evaluated BOP runner (`examples/bop_eval.py`) does extra work this
    class does not (weight sweep, disk cache, multi-instance, resume). This
    is the library wiring: segment → encode → solve → refine → score → select.
    """
    segmentor: Segmentor
    query_encoder: QueryEncoder
    target_encoder: TargetEncoder
    solver: PoseSolver
    refiners: Sequence[PoseRefiner]
    selector: Selector
    scorer: Optional[PoseScorer] = None
    topk: int = 2
    _query_cache: dict = field(default_factory=dict)

    def run(self, scene: Scene, obj: ObjectModel) -> Optional[PoseHypothesis]:
        qkey = (obj.obj_id, obj.mesh_path)
        q = self._query_cache.get(qkey)
        if q is None:
            with profiling.stage("query_encode(miss)"):
                q = self._query_cache[qkey] = self.query_encoder.encode_query(obj)
        frame = q.meta.get("canon_frame") or CanonFrame.from_points(q.pts)
        install = getattr(self.target_encoder, "install_pca", None)
        cands: list[PoseHypothesis] = []
        with profiling.stage("segment"):
            dets = self.segmentor.segment(scene, obj)[: self.topk]
        for det in dets:
            if install is not None:
                if "pca_vis" not in q.meta:
                    raise ValueError(
                        f"target encoder {type(self.target_encoder).__name__} "
                        f"exposes install_pca(), so it shares a per-object visual "
                        f"PCA with the query side — but the query features for "
                        f"obj_id={obj.obj_id} carry no 'pca_vis' snapshot to "
                        f"re-install. Re-encode the query through an encoder "
                        f"that snapshots its PCA.")
                install(q.meta["pca_vis"])
            t = self.target_encoder.encode_target(scene, det, obj, frame)
            with profiling.stage("solve"):
                hyps = list(self.solver.solve(q, t, frame))
            for h in hyps:
                for r in self.refiners:
                    with profiling.stage("refine"):
                        h = r.refine(h, scene, obj, q, t)
                if self.scorer is not None:
                    with profiling.stage("score"):
                        h = self.scorer.score(h, q, t)
                cands.append(h)
        return self.selector.select(cands)
