"""s_feat_w — the MATCHED-space feature score, recorded as a diagnostic.

`s_feat_1` scores in the canonical w=1 space on purpose (a weight-dependent
space systematically favours high-weight candidates when arbitrating across
weights). Claiming that choice was right needs the matched-space number the
frozen dumps never stored, and it cannot be reconstructed: the visual half is
rescaled BEFORE the cosine, so s_feat_w is not any function of s_feat_1 and w.

The load-bearing property here is that recording it changes NOTHING about the
selection — otherwise the diagnostic dump would not describe the frozen runs.
"""
import numpy as np
import pytest

pytest.importorskip("open3d")            # feature_aware_score -> pose_estimator

from popoe.interfaces import PointFeatures, PoseHypothesis
from popoe.scoring import ChampionScorer


def _clouds(seed=0, n=60, n_vis=4, n_geo=4, w=1.0):
    """A query/target pair whose visual half is scaled by w, with the unscaled
    pair kept in meta['feats_w1'] — the convention the weight sweep uses.

    The two sides carry DIFFERENT features (query = CAD side, target = scene
    side); identical features would make every cosine exactly 1 in both spaces
    and hide the very difference these tests are about.
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.05, 0.05, (n, 3))
    q_w1 = rng.standard_normal((n, n_vis + n_geo)).astype(np.float32)
    t_w1 = (q_w1 + 0.6 * rng.standard_normal(q_w1.shape)).astype(np.float32)

    def scale(f):
        s = f.copy()
        s[:, :n_vis] *= w
        return s

    q = PointFeatures(pts=pts, feats=scale(q_w1), meta={"feats_w1": q_w1})
    t = PointFeatures(pts=pts.copy(), feats=scale(t_w1), meta={"feats_w1": t_w1})
    return q, t


def _pose(s_icp=0.8):
    return PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                          breakdown={"s_icp": s_icp})


def test_off_by_default_and_absent():
    out = ChampionScorer().score(_pose(), *_clouds(w=0.3))
    assert "s_feat_w" not in out.breakdown


def test_recording_does_not_change_the_score():
    """The whole point: a dump run must arbitrate exactly like the frozen run."""
    q, t = _clouds(w=0.3)
    on = ChampionScorer(compute_s_feat_w=True).score(_pose(), q, t)
    off = ChampionScorer().score(_pose(), q, t)
    assert on.score == off.score                      # exact, not approx
    assert on.breakdown["s_feat_1"] == off.breakdown["s_feat_1"]
    assert (on.R == off.R).all() and (on.t == off.t).all()


def test_matched_space_differs_from_canonical_under_a_weight():
    """With w != 1 the two spaces are genuinely different measurements."""
    q, t = _clouds(w=0.2)
    bd = ChampionScorer(compute_s_feat_w=True).score(_pose(), q, t).breakdown
    assert "s_feat_w" in bd
    assert bd["s_feat_w"] != pytest.approx(bd["s_feat_1"], abs=1e-9)


def test_equal_when_no_weight_sweep_is_active():
    """No feats_w1 in meta -> both scores read the same array. Recording the
    equality is correct; it is a measurement, not a missing value."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.05, 0.05, (40, 3))
    feats = rng.standard_normal((40, 8)).astype(np.float32)
    q = PointFeatures(pts=pts, feats=feats, meta={})
    t = PointFeatures(pts=pts.copy(), feats=feats.copy(), meta={})
    bd = ChampionScorer(compute_s_feat_w=True).score(_pose(), q, t).breakdown
    assert bd["s_feat_w"] == pytest.approx(bd["s_feat_1"])


def test_composes_with_s_coarse_and_still_leaves_score_alone():
    q, t = _clouds(w=0.5)
    pose = PoseHypothesis(
        R=np.eye(3), t=np.zeros(3), score=0.0,
        breakdown={"s_icp": 0.8, "R_coarse": np.eye(3),
                   "t_coarse": np.array([10.0, 10.0, 10.0])})
    both = ChampionScorer(compute_s_coarse=True,
                          compute_s_feat_w=True).score(pose, q, t)
    coarse_only = ChampionScorer(compute_s_coarse=True).score(pose, q, t)
    assert both.score == coarse_only.score
    assert {"s_coarse", "s_feat_w"} <= both.breakdown.keys()


def test_recipes_wire_the_flag_through():
    from popoe.freeze.recipes import stages_for_object
    _, _, scorer = stages_for_object(0.1, score_feat_w=True)
    assert scorer.compute_s_feat_w is True
    _, _, scorer = stages_for_object(0.1)
    assert scorer.compute_s_feat_w is False
