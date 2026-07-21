"""ChampionScorer — the evaluated-best selection rule as a PoseScorer stage.

score = s_icp * max(s_feat_1, 0) * (metric_fit if size_aware else 1)

Where the terms come from (all measured, see the reproduction study):

  * ``s_icp`` — ICP inlier fitness (geometric agreement), from the refiner.
  * ``s_feat_1`` — mean feature cosine over inliers computed with the fused
    features at **visual weight 1** (the canonical, weight-invariant space).
    Scoring in a weight-dependent space systematically favours high-weight
    candidates when arbitrating across weights; the w=1 space fixes that. A
    26-rule exponent-grid ablation found ``icp * sfeat1`` (exponent ratio 1)
    exactly optimal, and the canonical-space family dominated the
    matched-space family everywhere.
  * ``metric_fit`` — bidirectional absolute-scale inlier fraction
    ``min(fit(query->target), fit(target->query))`` at ``size_thr`` metres.
    Canonical/fused scores are scale-blind, so a same-shape wrong-size
    candidate (a pooled confusable pair, e.g. the YCB-V clamps) looks perfect
    to them; at metre scale the size mismatch collapses this term. min() is
    required: one-directional fitness is blind to small-model-on-big-instance
    (every query point still finds a neighbour) and was measured to make the
    swap WORSE (68% -> 79%). Enable only for objects served from a pooled
    candidate set (``size_aware=True``); for ordinary objects it correlates
    with s_icp and just adds noise.

Feature convention: PointFeatures.feats fused as [vis | geo] halves. If a
runner sweeps visual weights by rescaling the vis half, it should pass the
UNSCALED (w=1) features in ``meta["feats_w1"]`` on both sides; the scorer
falls back to ``.feats`` when absent.
"""

from __future__ import annotations

import numpy as np

from popoe.interfaces import PointFeatures, PoseHypothesis


class ChampionScorer:
    """The paper's S_coarse (pre-ICP feature score, canonical w=1) can be either
    RECORDED as a diagnostic or USED as an arbitration factor:

        compute_s_coarse: record ``breakdown["s_coarse"]`` (S_coarse — the same
            feature_aware_score as s_feat_1 but at the PRE-ICP coarse pose, in
            the SAME canonical w=1 space) WITHOUT changing the score. Diagnostic.
        use_s_coarse: multiply the final score by ``max(s_coarse, 0)`` — i.e.
            arbitrate with ``s_icp * s_feat_1 * metric_fit * s_coarse``,
            byte-identical to rule_replay's rule of that name (all evidence
            clamped at 0; s_icp/metric_fit are already >= 0). Implies recording.

    Either needs the coarse pose in the breakdown (``R_coarse``/``t_coarse`` —
    set ICPRefiner(keep_coarse=True)); a missing coarse pose is a loud error,
    never a silent skip. With BOTH off the scorer is byte-identical to before
    (``s_icp * s_feat_1 * metric_fit``). S_coarse HELPS YCB-V (+2.5 in replay)
    but HURTS LM-O (-1.9): the 26-rule ablation shows rules do not transfer, so
    this is a per-DATASET switch (freeze.recipes.stages_for_object / bop_eval), not a
    hard-coded default — the first formal carrier of a per-dataset rule."""

    def __init__(self, tau_inlier_frac: float = 0.03, size_thr: float = 0.0075,
                 size_aware: bool = False, compute_s_coarse: bool = False,
                 use_s_coarse: bool = False):
        self.tau_inlier_frac = tau_inlier_frac      # fraction of query extent
        self.size_thr = size_thr                    # metres, metric_fit inliers
        self.size_aware = size_aware
        self.compute_s_coarse = compute_s_coarse
        self.use_s_coarse = use_s_coarse

    def score(self, pose: PoseHypothesis,
              query: PointFeatures, target: PointFeatures) -> PoseHypothesis:
        from popoe.registration import feature_aware_score

        fq = query.meta.get("feats_w1", query.feats)
        ft = target.meta.get("feats_w1", target.feats)
        diam = float(np.ptp(query.pts, axis=0).max())
        tau = self.tau_inlier_frac * diam
        s1, _ = feature_aware_score(
            pose.R, pose.t, query.pts, target.pts, fq, ft, tau)
        s_icp = pose.breakdown.get("s_icp", pose.breakdown.get("fitness", 0.0))

        met = 1.0
        if self.size_aware:
            met = self._metric_fit(pose, query.pts, target.pts)

        extra = {}
        sc_factor = 1.0
        if self.compute_s_coarse or self.use_s_coarse:
            if not {"R_coarse", "t_coarse"} <= pose.breakdown.keys():
                raise ValueError(
                    "s_coarse needs the coarse pose (R_coarse AND t_coarse) in "
                    "the breakdown; set ICPRefiner(keep_coarse=True)")
            # Same formula and canonical w=1 space as s_feat_1, at the PRE-ICP
            # pose -> the paper's S_coarse.
            sc, _ = feature_aware_score(
                pose.breakdown["R_coarse"], pose.breakdown["t_coarse"],
                query.pts, target.pts, fq, ft, tau)
            extra["s_coarse"] = float(sc)
            if self.use_s_coarse:
                sc_factor = max(float(sc), 0.0)     # arbitration factor

        score = float(s_icp) * max(float(s1), 0.0) * float(met) * sc_factor
        return PoseHypothesis(
            R=pose.R, t=pose.t, score=score,
            breakdown={**pose.breakdown, "s_feat_1": float(s1),
                       "metric_fit": float(met), **extra})

    def _metric_fit(self, pose: PoseHypothesis,
                    pts_q: np.ndarray, pts_t: np.ndarray) -> float:
        import open3d as o3d
        reg = o3d.pipelines.registration
        posed = pts_q @ pose.R.T + pose.t
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(posed.astype(np.float64))
        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(pts_t.astype(np.float64))
        eye = np.eye(4)
        return min(reg.evaluate_registration(src, tgt, self.size_thr, eye).fitness,
                   reg.evaluate_registration(tgt, src, self.size_thr, eye).fitness)
