"""
popoe.adapters — method-agnostic stage adapters. Thin wrappers that make the
generic registration primitives (popoe.registration) satisfy the stage
Protocols in popoe.interfaces, so they compose in `interfaces.Pipeline`.
Everything here is pure numpy+open3d and unit-testable offline.

The FreeZe-specific stages (encoders, FreeZeScorer, make_freeze_encoders)
live in popoe.freeze.adapters; they are re-exported at the bottom for
backwards compatibility.

One design point worth knowing: `ICPRefiner` moves geometry only; the final
feature scoring lives in the separate scorer stage (see interfaces.PoseScorer,
e.g. freeze.FreeZeScorer / scoring.ChampionScorer). `refine` still takes
`query` because ICP aligns the query point cloud (geometry), not for scoring.
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


# ── Pose solve / refine / select ────────────────────────────────────────

class RansacSolver:
    """Adapt ransac_pose_estimation -> one coarse PoseHypothesis (s_coarse)."""

    def __init__(self, n_ransac: int = 10000, tau_inlier: float = 0.03, k: int = 10):
        self.n_ransac = n_ransac
        self.tau_inlier = tau_inlier
        self.k = k

    def solve(self, query: PointFeatures, target: PointFeatures,
              frame: CanonFrame) -> list[PoseHypothesis]:
        from popoe.registration import ransac_pose_estimation
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
    (s_coarse) is carried through untouched for FreeZeScorer to finalise.

    `keep_coarse=True` stashes the PRE-ICP pose in the breakdown
    (``R_coarse`` / ``t_coarse``) so a scorer can evaluate the paper's S_coarse
    (a feature score at the coarse pose). Off by default — the stash is the only
    breakdown difference, so the refiner is byte-identical when it is off."""

    def __init__(self, tau_icp: float = 0.03, keep_coarse: bool = False):
        self.tau_icp = tau_icp
        self.keep_coarse = keep_coarse

    def refine(self, pose: PoseHypothesis, scene: Scene, obj: ObjectModel,
               query: PointFeatures, target: PointFeatures) -> PoseHypothesis:
        from popoe.registration import icp_refinement
        dense = target.pts_dense if target.pts_dense is not None else target.pts
        R_f, t_f, s_icp = icp_refinement(query.pts, dense, pose.R, pose.t, self.tau_icp)
        extra = {"R_coarse": pose.R, "t_coarse": pose.t} if self.keep_coarse else {}
        return PoseHypothesis(
            R=R_f, t=t_f, score=pose.score,     # provisional; FreeZeScorer sets final
            breakdown={**pose.breakdown, "s_icp": s_icp, "fitness": s_icp, **extra},
        )


class BestScoreSelector:
    """Pick the highest-scoring hypothesis (the multi-mask top-K choice)."""

    def select(self, candidates: list[PoseHypothesis]):
        cands = [c for c in candidates if c is not None]
        return max(cands, key=lambda h: h.score) if cands else None


def resolve_resume(row_stats: dict, target_counts: dict) -> tuple:
    """Classify already-written eval targets for resume, by ROW COUNT alone.

    Relies on the writer's completion invariant (examples/bop_eval.py): a
    finished target emits EXACTLY inst_count rows, zero-padded when fewer
    champions were found. Row contents are deliberately not consulted —
    "crashed after two rows" and "completed with two champions" are
    indistinguishable from contents, and a real score can format as 0.000000.

    Args:
        row_stats: {(scene, im, obj): n_rows} from the existing CSV.
        target_counts: {(scene, im, obj): inst_count} for this run's targets.

    Returns (done, partial):
        done    — n_rows >= inst_count: skip.
        partial — 0 < n_rows < inst_count: crash mid-target. Stale rows must
            be dropped from the CSV before re-running, or the rerun appends
            duplicates.

    With inst_count == 1 everywhere (LMO / YCB-V) any existing row marks its
    target done and partial is empty — identical to the old any-row rule."""
    done, partial = set(), set()
    for key, n_rows in row_stats.items():
        if n_rows <= 0:
            continue
        if n_rows >= target_counts.get(key, 1):
            done.add(key)
        else:
            partial.add(key)
    return done, partial


def select_top_instances(hyps_by_det: dict, selector, k: int) -> list:
    """BOP multi-instance selection: one champion per detection, then the top-k
    champions across detections.

    A detection is one candidate INSTANCE, so hypotheses within a detection are
    alternatives (pick one champion via `selector`), while champions of
    different detections are candidate distinct instances (keep up to k, best
    first — k comes from the BOP target's ``inst_count``). With k=1 this is
    exactly the old global argmax: max over per-detection maxima."""
    champs = [selector.select(hs) for hs in hyps_by_det.values()]
    champs = [c for c in champs if c is not None]
    champs.sort(key=lambda c: -c.score)
    return champs[:k]


# ── Backwards compatibility ─────────────────────────────────────────────
# The FreeZe-specific adapters moved to popoe.freeze.adapters; old import
# paths keep working.
from popoe.freeze.adapters import (  # noqa: E402,F401
    FreeZeQueryEncoder, FreeZeTargetEncoder, FreeZeScorer, make_freeze_encoders,
)
