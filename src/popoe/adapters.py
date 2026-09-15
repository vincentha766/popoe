"""
popoe.adapters — method-agnostic stage adapters. Thin wrappers that make the
generic registration primitives (popoe.registration) satisfy the stage
Protocols in popoe.interfaces. Pure numpy+open3d, unit-testable offline.

`ICPRefiner` moves geometry only; scoring is PoseScorer
(FreeZeScorer / ChampionScorer). It is both a correspondence ``PoseRefiner``
and a ``GeometricRefiner`` for ``DirectPoseMethod``. RansacSolver +
BestScoreSelector compose with them in `interfaces.Pipeline`.
"""

from __future__ import annotations
import math

import numpy as np

from popoe.interfaces import (
    Scene, ObjectModel, CanonFrame, PointFeatures, PoseHypothesis,
)


class RansacSolver:
    """Adapt ransac_pose_estimation -> one coarse PoseHypothesis (s_coarse)."""

    def __init__(self, n_ransac: int = 10000, tau_inlier: float = 0.03, k: int = 10):
        self.n_ransac = n_ransac
        self.tau_inlier = tau_inlier
        self.k = k

    def solve(self, query: PointFeatures, target: PointFeatures,
              frame: CanonFrame | None = None) -> list[PoseHypothesis]:
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
    compute the feature score (that is ChampionScorer's job). The provisional score
    (s_coarse) is carried through untouched for ChampionScorer to finalise.

    `keep_coarse=True` stashes the PRE-ICP pose in the breakdown
    (``R_coarse`` / ``t_coarse``) so a scorer can evaluate the paper's S_coarse
    (a feature score at the coarse pose). Off by default — the stash is the only
    breakdown difference, so the refiner is byte-identical when it is off."""

    def __init__(self, tau_icp: float = 0.03, keep_coarse: bool = False):
        self.tau_icp = tau_icp
        self.keep_coarse = keep_coarse

    def refine_geometry(self, pose: PoseHypothesis, scene: Scene,
                        obj: ObjectModel, pts_src: np.ndarray,
                        pts_tgt: np.ndarray) -> PoseHypothesis:
        """``GeometricRefiner``: ICP on two clouds, no descriptors."""
        del scene, obj
        from popoe.registration import icp_refinement
        R_f, t_f, s_icp = icp_refinement(pts_src, pts_tgt, pose.R, pose.t,
                                         self.tau_icp)
        extra = {"R_coarse": pose.R, "t_coarse": pose.t} if self.keep_coarse else {}
        return PoseHypothesis(
            R=R_f, t=t_f, score=pose.score,     # provisional; ChampionScorer sets final
            breakdown={**pose.breakdown, "s_icp": s_icp, "fitness": s_icp,
                       # Absolute, in the clouds' units (metres). A later stage
                       # that re-runs ICP must reuse THIS tau or its fitness is
                       # on a different scale than ours, and s_icp feeds the
                       # scorer directly. Guessing a default silently meant
                       # 30 mm (see render_rerank._tau_icp).
                       "tau_icp": float(self.tau_icp), **extra},
        )

    def refine(self, pose: PoseHypothesis, scene: Scene, obj: ObjectModel,
               query: PointFeatures, target: PointFeatures) -> PoseHypothesis:
        """``PoseRefiner``: same ICP, clouds taken from encoded features."""
        dense = target.pts_dense if target.pts_dense is not None else target.pts
        return self.refine_geometry(pose, scene, obj, query.pts, dense)


def best_hyp(candidates):
    cands = [c for c in candidates if c is not None]
    return max(cands, key=lambda h: h.score) if cands else None


class BestScoreSelector:
    """Pick the highest-scoring hypothesis (the multi-mask top-K choice)."""

    def select(self, candidates: list[PoseHypothesis]):
        return best_hyp(candidates)


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


def depth_mask_cloud(scene: Scene, mask=None, max_pts: int = 3000):
    """Back-project ``(depth > 0)`` pixels (optionally inside ``mask``) to
    metres in the camera frame.

    Same construction as the GeDi neighbourhood cloud and as
    ``examples.bop_eval.dense_mask_cloud``. ``max_pts`` 0 = no cap; otherwise
    a fixed-seed draw via :func:`fixed_seed_subsample`. Returns ``None``
    when fewer than 4 valid pixels (ICP then has no target cloud).
    """
    depth = np.asarray(scene.depth)
    valid = depth > 0
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)
    ys, xs = np.where(valid)
    if len(ys) < 4:
        return None
    idx = fixed_seed_subsample(len(ys), max_pts)
    if idx is not None:
        ys, xs = ys[idx], xs[idx]
    d = depth[ys, xs]
    fx, fy = float(scene.K[0, 0]), float(scene.K[1, 1])
    cx, cy = float(scene.K[0, 2]), float(scene.K[1, 2])
    return np.stack([(xs - cx) * d / fx, (ys - cy) * d / fy, d],
                    axis=1).astype(np.float32)


def cad_surface_cloud(obj: ObjectModel, n_points: int = 3000,
                      seed: int | None = None) -> np.ndarray:
    """CAD surface sample in **metres**.

    Sampling is :func:`popoe.freeze.adapters.sample_query_surface` (mesh
    units — millimetres on BOP). ``ObjectModel.diameter`` is metres; if the
    sampled extent is more than 10× the diameter the cloud is treated as mm
    and divided by 1000. Metre meshes (extent already on the diameter
    scale) are left unchanged.
    """
    from popoe.freeze.adapters import sample_query_surface
    seed = obj.obj_id if seed is None else int(seed)
    pts = np.asarray(sample_query_surface(obj.mesh_path, n_points, seed),
                     dtype=np.float64)
    extent = float(np.ptp(pts, axis=0).max()) if pts.size else 0.0
    diam = float(obj.diameter)
    if diam > 0 and extent > 10.0 * diam:
        pts = pts / 1000.0
    return pts.astype(np.float32)


def icp_clouds(scene: Scene, obj: ObjectModel, det=None, *,
               n_cad: int = 3000, max_scene: int = 3000):
    """Default ``DirectPoseMethod.clouds``: CAD (m) and scene (m).

    ``det.mask`` crops the scene cloud when present; otherwise every valid
    depth pixel is used. Raises if the scene cloud is degenerate.
    """
    mask = None if det is None else getattr(det, "mask", None)
    pts_tgt = depth_mask_cloud(scene, mask, max_pts=max_scene)
    if pts_tgt is None:
        raise ValueError(
            "icp_clouds: fewer than 4 valid depth pixels "
            f"(obj_id={obj.obj_id})")
    pts_src = cad_surface_cloud(obj, n_cad)
    return pts_src, pts_tgt


def fixed_seed_subsample(n: int, cap: int):
    """Sorted fixed-seed uniform draw of ``cap`` of ``n`` indices; None = keep
    all. THE single source for P_T^dense subsampling: the paper's Sec. IV-A
    dense target cloud is ONE 3k cloud serving both the GeDi neighbourhood
    (feature_extractor, POPOE_TARGET_DENSE) and ICP (bop_eval's
    dense_mask_cloud). Both sides draw through this function over the same
    row-major (depth>0)&mask index space, so they reach the identical cloud
    without threading it through the feature cache — duplicate the rule and
    the two stages no longer share one cloud."""
    if cap and n > cap:
        return np.sort(np.random.default_rng(0).choice(n, cap, replace=False))
    return None


def query_camera_radius(extent: float, fill: float, mode: str,
                        fov_deg: float = 60.0) -> float:
    """Camera orbit radius for the query renders (audit P2).

    ``legacy`` (default): ``1.5 * extent`` — the historical code, where the
    fill knob is INERT: the mesh is pre-scaled by ``fill * canon / extent``
    and the radius is taken from the SCALED extent, so fill cancels out of
    the projection and the object always spans ~1/(1.5 * 2*tan(fov/2)) ≈ 0.58
    of the 60-degree frame regardless of POPOE_QUERY_FILL. Kept byte-identical
    because every historical cache key was built on it.

    ``effective``: the radius that makes the object's largest side span
    ``fill`` of the canvas in the weak-perspective (central-chord) sense:
    ``extent / (fill * 2 * tan(fov/2))``. Measured silhouettes deviate by
    perspective (sphere 0.52, box median 0.50 / max 0.56 at fill=0.5) — the
    paper's "approximately 50%" becomes a real, approximately-honoured
    setting instead of a no-op. Gated behind POPOE_QUERY_FILL_MODE (an
    enc_cfg conditional key) because making fill effective changes the
    renders under otherwise-unchanged cache keys. fill above 0.7 is refused:
    by 0.9 the measured silhouette already exits the 60-degree frame (~1.05)
    and clipped u/v silently sample edge pixels in the depth test."""
    if mode == "legacy":
        return 1.5 * extent
    if mode == "effective":
        if not 0.0 < fill <= 0.7:
            raise ValueError(
                f"effective fill must be in (0, 0.7], got {fill} — larger "
                f"values push the silhouette out of the 60-degree frame")
        return extent / (fill * 2.0 * math.tan(math.radians(fov_deg) / 2.0))
    raise ValueError(
        f"POPOE_QUERY_FILL_MODE must be legacy|effective, got {mode!r}")


def paper_grid_centers(y0: int, y1: int, x0: int, x1: int, grid_size: int):
    """Patch centres of the minimal axis-aligned SQUARE bbox containing the
    mask bbox [y0..y1] x [x0..x1] (paper Sec. III-D target protocol, triage
    D3): the square's side is the mask bbox's larger side, centred on it; the
    centres are a grid_size x grid_size tiling — centre of tile (i, j) sits at
    (i+0.5, j+0.5) / grid_size of the square. Returns (rows, cols, u, v,
    (bx0, by0, side)) with u/v rounded to ints, UNFILTERED — the caller drops
    centres outside the image / mask / valid depth. rows/cols index straight
    into the grid_size x grid_size DINOv2 patch feature map of the square
    crop resized to grid_size*14 (direct patch assignment, no bilinear)."""
    # The square is SNAPPED to the integer pixel grid (floor): the DINO crop
    # and this tiling must share EXACTLY the same box. Rounding the crop
    # independently (review F1) shifted it by 0.5 px against the tiling
    # whenever the padded axis had odd slack — up to a full patch of feature
    # misassignment (worst case: every centre off by one tile). side is
    # integer-valued for integer bboxes, so flooring the corner pins the
    # whole box to integers and the crop below reproduces it exactly.
    side = float(max(y1 - y0, x1 - x0) + 1)
    by0 = float(np.floor((y0 + y1 + 1) / 2.0 - side / 2.0))
    bx0 = float(np.floor((x0 + x1 + 1) / 2.0 - side / 2.0))
    rows, cols = np.meshgrid(np.arange(grid_size), np.arange(grid_size),
                             indexing="ij")
    rows = rows.reshape(-1)
    cols = cols.reshape(-1)
    # floor, not round: pixel k covers the continuous span [k, k+1), so the
    # pixel CONTAINING a centre is floor(centre). np.round's half-to-even
    # would duplicate pixels and step outside the box whenever centres land
    # exactly on .5 (every integer side/grid ratio does).
    u = np.floor(bx0 + (cols + 0.5) * side / grid_size).astype(int)
    v = np.floor(by0 + (rows + 0.5) * side / grid_size).astype(int)
    return rows, cols, u, v, (bx0, by0, side)


def select_top_instances(hyps_by_det: dict, k: int,
                         nms_dist: float = 0.0) -> list:
    """BOP multi-instance selection: one champion per detection, translation
    NMS across champions, then the top-k.

    A detection is one candidate INSTANCE, so hypotheses within a detection are
    alternatives (pick one champion), while champions of different detections
    are candidate distinct instances (keep up to k, best first — k comes from
    the BOP target's ``inst_count``). With k=1 this is the old global argmax.

    ``nms_dist`` (metres, 0 = off) is FreeZeV2 §III-F's duplicate removal.
    A multi-source union deliberately keeps every segmentor's mask, so one
    physical instance can win 2-4 detections with near-identical poses and
    occupy that many of the k slots — the paper's protocol resolves this at
    the END, on refined poses, by NMS over translation distance, not by
    filtering masks up front. Greedy best-first: keep a champion only if its
    translation is at least ``nms_dist`` from every kept one. Runs BEFORE the
    top-k cut so a suppressed duplicate's slot goes to the next distinct
    instance; survivors short of k are NOT padded with suppressed duplicates
    (the paper retains "only distinct object instance poses"). The paper
    names the mechanism but no radius — callers derive one from the object
    diameter (see examples/bop_eval.py --trans-nms)."""
    champs = [best_hyp(hs) for hs in hyps_by_det.values()]
    champs = [c for c in champs if c is not None]
    champs.sort(key=lambda c: -c.score)
    if nms_dist > 0.0:
        # A champion with a non-finite translation cannot be distance-compared:
        # NaN >= nms_dist is False against EVERY later champion, so one garbage
        # row would suppress the whole target. Such a pose is unusable anyway —
        # drop it before the greedy pass.
        champs = [c for c in champs if np.isfinite(c.t).all()]
        kept = []
        for c in champs:
            if all(float(np.linalg.norm(c.t - p.t)) >= nms_dist for p in kept):
                kept.append(c)
        champs = kept
    return champs[:k]
