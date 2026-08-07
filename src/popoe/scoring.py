"""ChampionScorer — the evaluated-best selection rule as a PoseScorer stage.

score = s_icp * max(s_feat_1, 0) * (metric_fit if size_aware else 1)

Where the terms come from (all measured, see the reproduction study):

  * ``s_icp`` — ICP inlier fitness (geometric agreement), from the refiner.
  * ``s_feat_1`` — mean feature cosine over inliers computed with the fused
    features at **visual weight 1** (the canonical, weight-invariant space).
    Scoring in a weight-dependent space biases the arbitration across weights;
    the w=1 space fixes that. A 26-rule exponent grid over both spaces
    (2026-08-04, frozen `tuned-cnos` dumps with the diagnostic ``s_feat_w``
    column) has the canonical family ahead at **all 13 exponent pairs on both
    datasets**: best-vs-best +0.74 pt on LM-O, +2.95 pt on YCB-V.

    The bias runs toward LOW weights, not high ones. Measured on the same
    dumps: under the matched-space rule 57% (LM-O) / 46% (YCB-V) of targets
    pick w=0.2, mean picked w drops 0.577->0.355 and 0.631->0.441, and only
    ~38% of targets keep the same champion. Mechanism: at small w the visual
    half is shrunk, so the fused vector is dominated by the geometric half,
    whose cosines agree more readily — the matched score rises as w falls
    (median s_feat_w 0.2867->0.4154 on LM-O as w goes 1.0->0.2, while s_feat_1
    stays flat). An earlier version of this note said the bias favours
    high-weight candidates; that direction was never measured and is wrong.

    Note also that ratio 1 is NOT the grid optimum on these dumps (LM-O
    ``s_icp^1.5*s_feat_1`` 0.7119 vs 0.7069 at (1,1); YCB-V 0.8277 vs 0.8265).
    The live rule is kept at ratio 1 deliberately: picking a maximum out of 13
    post-hoc rules is a multiple-comparison trap, and the spread among the top
    canonical rules is under 0.5 pt.
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
        compute_s_feat_w: record ``breakdown["s_feat_w"]`` — the same score at
            the SAME (post-ICP) pose but in the MATCHED, weight-scaled space,
            i.e. the space s_feat_1 deliberately does not use. Diagnostic only;
            the score is untouched. This is what a canonical-vs-matched
            comparison needs and what the campaign2-era dumps lack.
        use_s_coarse: multiply the final score by ``max(s_coarse, 0)`` — i.e.
            arbitrate with ``s_icp * s_feat_1 * metric_fit * s_coarse``,
            byte-identical to rule_replay's rule of that name (all evidence
            clamped at 0; s_icp/metric_fit are already >= 0). Implies recording.
        use_render_score: multiply the final score by ``max(sar_ti, 0)`` — the
            input-vs-render appearance score the rerank stage left in the
            breakdown (v2.1's "compare input image with the rendered pose"
            component, decision 13). Requires ``--render-rerank`` upstream.

    Either needs the coarse pose in the breakdown (``R_coarse``/``t_coarse`` —
    set ICPRefiner(keep_coarse=True)); a missing coarse pose is a loud error,
    never a silent skip. With BOTH off the scorer is byte-identical to before
    (``s_icp * s_feat_1 * metric_fit``). S_coarse HELPS YCB-V (+2.5 in replay)
    but HURTS LM-O (-1.9): the 26-rule ablation shows rules do not transfer, so
    this is a per-DATASET switch (freeze.recipes.stages_for_object / bop_eval), not a
    hard-coded default — the first formal carrier of a per-dataset rule."""

    def __init__(self, tau_inlier_frac: float = 0.03, size_thr: float = 0.0075,
                 size_aware: bool = False, compute_s_coarse: bool = False,
                 use_s_coarse: bool = False, tau_abs: float | None = None,
                 compute_s_feat_w: bool = False, eq5_terms: bool = False,
                 use_render_score: bool = False):
        self.tau_inlier_frac = tau_inlier_frac      # fraction of query extent
        # Absolute tau_inlier in metres, overriding tau_inlier_frac * extent.
        # FreeZeV2 Sec. IV-A: "Thresholds tau_inlier and tau_ICP are set to 3% of
        # the object's DIAMETER" — the sampled query cloud's bounding-box side is
        # a different (and systematically smaller) quantity, so a caller that
        # knows the BOP models_info diameter passes the product here instead.
        # None keeps the historical fraction-of-extent behaviour byte-identical.
        self.tau_abs = tau_abs
        self.size_thr = size_thr                    # metres, metric_fit inliers
        self.size_aware = size_aware
        self.compute_s_coarse = compute_s_coarse
        self.use_s_coarse = use_s_coarse
        # Diagnostic-only: the SAME feature score in the MATCHED (weight-scaled)
        # space that s_feat_1 measures in the canonical w=1 space. Never enters
        # `score` — it exists so a canonical-vs-matched rule family can be
        # compared on one frozen candidate pool instead of being asserted.
        self.compute_s_feat_w = compute_s_feat_w
        # Paper Eq.7 form (faithful arms): both feature terms use the Eq.5
        # formulation — target->query top-k=10 feature pool, tau_inlier
        # inliers, FIXED |P_T^sparse| denominator — instead of the historical
        # mean-cosine-over-geometric-NN. Sec. III-F: the fine score is
        # recomputed "using the same formulation provided in Eq. (5)". Off by
        # default: the tuned scoring identity stays byte-identical.
        self.eq5_terms = eq5_terms
        # v2.1 third component (decision 13): multiply the final score by the
        # clamped input-vs-render DINOv2 appearance score (`sar_ti`) of the
        # variant the rerank stage kept — the paper only says v2.1 "compares
        # the visual features of the input image with the rendered pose" to
        # improve scoring; this factor form is pinned-by-us (same clamped
        # arbitration shape as use_s_coarse). The render pass is the rerank
        # stage's own, so this adds ZERO renders; a chain without that stage
        # is a wiring error (loud), not a degraded mode. Off by default: both
        # arms' pre-decision-13 identities stay byte-identical.
        self.use_render_score = use_render_score

    def score(self, pose: PoseHypothesis,
              query: PointFeatures, target: PointFeatures) -> PoseHypothesis:
        from popoe.registration import feature_aware_score, eq5_score
        score_fn = eq5_score if self.eq5_terms else feature_aware_score

        fq = query.meta.get("feats_w1", query.feats)
        ft = target.meta.get("feats_w1", target.feats)
        diam = float(np.ptp(query.pts, axis=0).max())
        tau = self.tau_abs if self.tau_abs is not None else self.tau_inlier_frac * diam
        s1, _ = score_fn(
            pose.R, pose.t, query.pts, target.pts, fq, ft, tau)
        s_icp = pose.breakdown.get("s_icp", pose.breakdown.get("fitness", 0.0))

        met = 1.0
        if self.size_aware:
            met = self._metric_fit(pose, query.pts, target.pts)

        extra = {}
        if self.compute_s_feat_w:
            # query.feats / target.feats are the WEIGHT-SCALED features the
            # runner produced for this candidate's w; feats_w1 (used above) is
            # the unscaled pair. When no weight sweep is active the two are the
            # same array and s_feat_w == s_feat_1 — a real equality, not a
            # missing measurement, so it is still recorded.
            sw, _ = score_fn(
                pose.R, pose.t, query.pts, target.pts,
                query.feats, target.feats, tau)
            extra["s_feat_w"] = float(sw)

        sc_factor = 1.0
        if self.compute_s_coarse or self.use_s_coarse:
            if not {"R_coarse", "t_coarse"} <= pose.breakdown.keys():
                raise ValueError(
                    "s_coarse needs the coarse pose (R_coarse AND t_coarse) in "
                    "the breakdown; set ICPRefiner(keep_coarse=True)")
            # Same formula and canonical w=1 space as s_feat_1, at the PRE-ICP
            # pose -> the paper's S_coarse.
            sc, _ = score_fn(
                pose.breakdown["R_coarse"], pose.breakdown["t_coarse"],
                query.pts, target.pts, fq, ft, tau)
            extra["s_coarse"] = float(sc)
            if self.use_s_coarse:
                sc_factor = max(float(sc), 0.0)     # arbitration factor

        rs_factor = 1.0
        if self.use_render_score:
            if "sar_ti" in pose.breakdown:
                rs_factor = max(float(pose.breakdown["sar_ti"]), 0.0)
            elif pose.breakdown.get("render_rerank") == "skipped_no_bbox":
                # The rerank stage ran but had no bbox to render against (its
                # one explicit skip path; unreachable from bop_eval, where
                # every target is built from a detection). Neutral factor,
                # and the marker is already in the breakdown.
                pass
            else:
                raise ValueError(
                    "use_render_score needs sar_ti in the breakdown; wire "
                    "RenderAppearanceReranker (--render-rerank) before the "
                    "scorer")

        score = (float(s_icp) * max(float(s1), 0.0) * float(met) * sc_factor
                 * rs_factor)
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
