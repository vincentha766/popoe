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
    def __init__(self, tau_inlier_frac: float = 0.03, size_thr: float = 0.0075,
                 size_aware: bool = False):
        self.tau_inlier_frac = tau_inlier_frac      # fraction of query extent
        self.size_thr = size_thr                    # metres, metric_fit inliers
        self.size_aware = size_aware

    def score(self, pose: PoseHypothesis,
              query: PointFeatures, target: PointFeatures) -> PoseHypothesis:
        from popoe.pose_estimator import feature_aware_score

        fq = query.meta.get("feats_w1", query.feats)
        ft = target.meta.get("feats_w1", target.feats)
        diam = float(np.ptp(query.pts, axis=0).max())
        s1, _ = feature_aware_score(
            pose.R, pose.t, query.pts, target.pts, fq, ft,
            self.tau_inlier_frac * diam)
        s_icp = pose.breakdown.get("s_icp", pose.breakdown.get("fitness", 0.0))

        met = 1.0
        if self.size_aware:
            met = self._metric_fit(pose, query.pts, target.pts)

        score = float(s_icp) * max(float(s1), 0.0) * float(met)
        return PoseHypothesis(
            R=pose.R, t=pose.t, score=score,
            breakdown={**pose.breakdown, "s_feat_1": float(s1),
                       "metric_fit": float(met)})

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
