"""The v2.1 render-vs-input component (decision 13): the rerank stage's
``sar_ti`` (input-vs-render DINOv2 appearance score, one per candidate) enters
champion selection as a clamped multiplicative factor
(``ChampionScorer(use_render_score=True)``). Off by default so both arms'
pre-decision-13 identities stay byte-identical. numpy + sklearn
(feature_aware_score); no GPU — the factor consumes a breakdown key the rerank
stage wrote, it never renders.
"""
import numpy as np
import pytest

# ChampionScorer.score() -> feature_aware_score -> pose_estimator, which hard-
# imports open3d at module load (reference extra). Skip cleanly without it.
pytest.importorskip("open3d")

from popoe.interfaces import PointFeatures, PoseHypothesis
from popoe.scoring import ChampionScorer


def _pf(pts, feats):
    return PointFeatures(pts=pts, feats=feats, meta={"feats_w1": feats})


def _identical_clouds(seed=0, n=60):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.05, 0.05, (n, 3))
    feats = rng.standard_normal((n, 8))
    return _pf(pts, feats), _pf(pts.copy(), feats.copy())


def test_off_is_byte_identical_even_with_sar_ti_present():
    """use_render_score=False (default): a candidate that went through the
    rerank stage (sar_ti in the breakdown) scores exactly as before."""
    q, t = _identical_clouds()
    pose = PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                          breakdown={"s_icp": 0.8, "sar_ti": 0.5})
    out = ChampionScorer(size_aware=False).score(pose, q, t)
    assert out.score == pytest.approx(0.8 * out.breakdown["s_feat_1"])


def test_on_multiplies_by_clamped_sar_ti():
    q, t = _identical_clouds()
    pose = PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                          breakdown={"s_icp": 0.8, "sar_ti": 0.5})
    out = ChampionScorer(size_aware=False, use_render_score=True).score(
        pose, q, t)
    assert out.score == pytest.approx(0.8 * out.breakdown["s_feat_1"] * 0.5)

    # A negative patch cosine is clamped at 0, like every other factor.
    neg = PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                         breakdown={"s_icp": 0.8, "sar_ti": -0.2})
    out_neg = ChampionScorer(size_aware=False, use_render_score=True).score(
        neg, q, t)
    assert out_neg.score == 0.0


def test_on_without_sar_ti_is_a_loud_wiring_error():
    """No rerank stage upstream -> no sar_ti -> refuse. A silent 1.0 here
    would make --render-score without --render-rerank an identity no-op."""
    q, t = _identical_clouds()
    pose = PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                          breakdown={"s_icp": 0.8})
    with pytest.raises(ValueError, match="sar_ti"):
        ChampionScorer(size_aware=False, use_render_score=True).score(
            pose, q, t)


def test_explicit_no_bbox_skip_is_neutral():
    """The rerank stage's one skip path (no bbox to render against) marks the
    breakdown; exactly that marker gets a neutral factor instead of the loud
    error — the stage ran, it just had nothing to render."""
    q, t = _identical_clouds()
    pose = PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                          breakdown={"s_icp": 0.8,
                                     "render_rerank": "skipped_no_bbox"})
    out = ChampionScorer(size_aware=False, use_render_score=True).score(
        pose, q, t)
    assert out.score == pytest.approx(0.8 * out.breakdown["s_feat_1"])


def test_stages_for_object_wires_and_guards():
    """render_score reaches the scorer; without render_rerank it is refused at
    construction (sar_ti would have no producer)."""
    from popoe.freeze.recipes import stages_for_object
    with pytest.raises(ValueError, match="render_rerank"):
        stages_for_object(0.1, render_score=True, render_rerank=False)
    _, _, scorer_off = stages_for_object(0.1)
    assert scorer_off.use_render_score is False
    _, _, scorer_on = stages_for_object(0.1, render_rerank=True,
                                        render_score=True)
    assert scorer_on.use_render_score is True
