"""
popoe.interfaces — the stage contracts.

This file is the *specification* layer: the data objects that flow between
stages, and the Protocol each swappable stage must satisfy. Nothing here does
heavy work; the reference implementations live in popoe.adapters,
popoe.freeze.feature_extractor, popoe.registration, popoe.solvers, ... and are wired
by `Pipeline` (see ARCHITECTURE.md).

Design goal: every stage below can be re-implemented independently (a new
segmentor, a new pose solver like TEASER++/MAC, a new fusion rule) without
touching the others, as long as it honours the input/output contract.

The Protocols use `typing.Protocol` (structural typing): an implementation does
NOT need to import or subclass anything here — it just needs matching method
signatures. This keeps implementations decoupled from the spec.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable
import numpy as np


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

    Substitution is a CALLER's policy: the caller composes an explicit chain
    (segmentor.FirstAvailableSegmentor) and can read back what ran.

    This is an *availability* signal, not an error channel. A runtime failure —
    CUDA OOM, a corrupt mesh — must propagate: "the fallback handled it" is how
    real bugs get buried."""


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
class ObjectModel:
    """A target object: CAD mesh + BOP metadata. The single source of truth for
    diameter and symmetry, so downstream stages never re-guess them."""
    obj_id: int
    mesh_path: str                      # BOP .ply, vertices in mm
    diameter: float                     # metres — drives CanonFrame.scale
    symmetries: Sequence[np.ndarray] = field(default_factory=list)  # 4x4 transforms


@dataclass(frozen=True)
class CanonFrame:
    """Canonicalisation convention shared by query & target so both live in the
    same registration space: pts_canon = (pts - center) * scale.

    IMPORTANT — this must match the live convention in
    popoe/freeze/feature_extractor.py exactly, or GeDi sees a different scale and
    poses change:
      * center = 0  (the current code does NOT centre — it applies pure scaling
        `pts * scale`; centring is kept in the contract for generality but is
        zero today).
      * scale  = 1 / max_extent, where max_extent is the largest side of the
        QUERY sampled point cloud's bounding box in metres
        (`np.ptp(pts, axis=0).max()`), NOT the BOP diameter. GeDi's r_lrf=0.5 m
        was trained on ~1 m scenes, so the object is rescaled to ~1 m extent.

    The scale is therefore computed by the query encoder from its sampled points
    and then REUSED on the target side (the `_canon_scale` side-channel today).
    """
    center: np.ndarray                  # (3,)
    scale: float

    @classmethod
    def from_points(cls, pts: np.ndarray) -> "CanonFrame":
        """Reproduce the live convention from a query point cloud (metres)."""
        extent_m = float(np.ptp(pts, axis=0).max())
        return cls(center=np.zeros(3, np.float32), scale=1.0 / max(extent_m, 1e-6))


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
    `feats` is already fused and L2-normed (see FeatureFusion)."""
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
# 3. The swappable stages. Implement any one to extend the pipeline.
# ════════════════════════════════════════════════════════════════════════

@runtime_checkable
class Segmentor(Protocol):
    """Stage 0. GT-mask / CNOS / SAM / multi-mask top-K all fit here."""
    def segment(self, scene: Scene, obj: ObjectModel) -> list[Detection]: ...


@runtime_checkable
class FeatureFusion(Protocol):
    """Combine per-point visual and geometric features into one descriptor.
    Reference impl: popoe.freeze.fusion.DinoGeDiFusion. Swap for concat-only,
    learned fusion, single-modality ablations, etc."""
    def fuse(self, vis_feats: np.ndarray, geo_feats: np.ndarray,
             apply_skip_vis: bool = False) -> np.ndarray: ...


@runtime_checkable
class QueryEncoder(Protocol):
    """Stage 1a. CAD -> sparse fused features (offline, cached per object).

    The CanonFrame is an OUTPUT here, not an input: its scale is derived from
    the sampled query points (see CanonFrame). Return it in
    PointFeatures.meta['canon_frame'] so the target side can reuse it."""
    def encode_query(self, obj: ObjectModel) -> PointFeatures: ...


@runtime_checkable
class TargetEncoder(Protocol):
    """Stage 1b. Masked RGB-D -> sparse fused features (online, per detection)."""
    def encode_target(self, scene: Scene, det: Detection,
                      obj: ObjectModel, frame: CanonFrame) -> PointFeatures: ...


@runtime_checkable
class PoseSolver(Protocol):
    """Stage 2. Feature matching -> coarse pose(s). RANSAC today; TEASER++ / MAC /
    consistency-graph are drop-in alternatives (the identified accuracy gap)."""
    def solve(self, query: PointFeatures, target: PointFeatures,
              frame: CanonFrame) -> list[PoseHypothesis]: ...


@runtime_checkable
class PoseRefiner(Protocol):
    """Stage 3. Move a hypothesis' geometry (and report a geometric fitness in
    breakdown), NOTHING about feature scoring. ICP and symmetry-refine are both
    Refiners and can be chained. `query` is here because ICP aligns the query
    point cloud to the target (query GEOMETRY), and sym-refine ignores it — a
    refiner uses whatever it needs. Final feature scoring is a separate PoseScorer
    stage (coupling point #3), so a new refiner never re-implements final_score."""
    def refine(self, pose: PoseHypothesis, scene: Scene, obj: ObjectModel,
               query: PointFeatures, target: PointFeatures) -> PoseHypothesis: ...


@runtime_checkable
class PoseScorer(Protocol):
    """Stage 3b. Assign the final score to a (refined) hypothesis. Owns the whole
    feature-scoring concern that used to be split across solver and refiner:
    fine re-score + the s_coarse/s_fine/s_icp combination. Swap to change the
    scoring rule (e.g. drop s_fine, reweight) without touching solve/refine."""
    def score(self, pose: PoseHypothesis,
              query: PointFeatures, target: PointFeatures) -> PoseHypothesis: ...


@runtime_checkable
class Selector(Protocol):
    """Choose the winning hypothesis across candidate masks / poses. The
    multi-mask top-K selection and adaptive-weight post-processing live here."""
    def select(self, candidates: list[PoseHypothesis]) -> Optional[PoseHypothesis]: ...


@runtime_checkable
class Metric(Protocol):
    """Stage 4. VSD / MSSD / MSPD / ADD(-S) / grasp. Returns named scores."""
    def compute(self, est: PoseHypothesis, gt: PoseHypothesis,
                obj: ObjectModel, scene: Scene) -> dict: ...


# Reference wiring — the control flow examples/ drive (see ARCHITECTURE.md).
@dataclass
class Pipeline:
    segmentor: Segmentor
    query_encoder: QueryEncoder
    target_encoder: TargetEncoder
    solver: PoseSolver
    refiners: Sequence[PoseRefiner]
    selector: Selector
    scorer: Optional[PoseScorer] = None   # None -> keep the score the solver/refiner set
    topk: int = 2
    _query_cache: dict = field(default_factory=dict)

    def run(self, scene: Scene, obj: ObjectModel) -> Optional[PoseHypothesis]:
        # Keyed by (obj_id, mesh_path): BOP object ids are only unique within
        # one dataset, and a Pipeline instance may be reused across two.
        qkey = (obj.obj_id, obj.mesh_path)
        q = self._query_cache.get(qkey)
        if q is None:
            q = self._query_cache[qkey] = self.query_encoder.encode_query(obj)
        # CanonFrame is produced by query encoding and reused on the target side.
        frame = q.meta.get("canon_frame") or CanonFrame.from_points(q.pts)
        cands: list[PoseHypothesis] = []
        for det in self.segmentor.segment(scene, obj)[: self.topk]:
            t = self.target_encoder.encode_target(scene, det, obj, frame)
            for h in self.solver.solve(q, t, frame):
                for r in self.refiners:
                    h = r.refine(h, scene, obj, q, t)     # geometry only
                if self.scorer is not None:
                    h = self.scorer.score(h, q, t)        # final feature score
                cands.append(h)
        return self.selector.select(cands)
