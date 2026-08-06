"""Eq.5-form scoring for the faithful arms (triage D1).

The paper's Eq.7 combines S_feat^coarse and S_feat^fine, BOTH "using the same
formulation provided in Eq. (5)": target->query top-k feature pool, tau_inlier
inliers, and the FIXED |P_T^sparse| denominator. The historical scorer used
mean cosine over geometric nearest neighbours — a different pool AND a
different normalisation. eq5_terms switches ChampionScorer to the paper form;
off must stay byte-identical (the tuned scoring identity).
"""
import numpy as np
import pytest

from popoe.interfaces import PoseHypothesis, PointFeatures
from popoe.registration import eq5_score, feature_aware_score
from popoe.scoring import ChampionScorer

I3, Z3 = np.eye(3), np.zeros(3)


def test_fixed_denominator_rewards_quantity():
    # Two sparse targets, only one covered within tau. Eq.5 divides by
    # |P_T| = 2; mean-cosine would divide by |I| = 1 and hide the miss.
    pts_q = np.array([[0.0, 0, 0]])
    pts_t = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    fq = np.array([[1.0, 0]], np.float32)
    ft = np.array([[1.0, 0], [1.0, 0]], np.float32)
    s, n = eq5_score(I3, Z3, pts_q, pts_t, fq, ft, 0.05, k=1)
    assert n == 1
    assert s == pytest.approx(0.5)


def test_pool_is_feature_topk_not_geometric_nn():
    # Query A matches the target's FEATURE but sits far away; query B sits
    # exactly on the target with a merely-similar feature. Eq.5's pool
    # (feature top-1) picks A -> outside tau -> score 0. The geometric-NN
    # mean cosine picks B -> 0.6. Same inputs, different verdicts — this is
    # the D1 divergence in miniature.
    pts_q = np.array([[5.0, 0, 0], [0.0, 0, 0]])
    pts_t = np.array([[0.0, 0, 0]])
    fq = np.array([[1.0, 0], [0.6, 0.8]], np.float32)
    ft = np.array([[1.0, 0]], np.float32)
    s5, n5 = eq5_score(I3, Z3, pts_q, pts_t, fq, ft, 0.05, k=1)
    assert (s5, n5) == (0.0, 0)
    sm, _ = feature_aware_score(I3, Z3, pts_q, pts_t, fq, ft, 0.05)
    assert sm == pytest.approx(0.6)


def test_one_target_can_contribute_k_inlier_pairs():
    pts_q = np.array([[0.0, 0, 0], [0.001, 0, 0]])
    pts_t = np.array([[0.0, 0, 0]])
    fq = np.array([[1.0, 0], [0.9, 0.1]], np.float32)
    ft = np.array([[1.0, 0]], np.float32)
    s, n = eq5_score(I3, Z3, pts_q, pts_t, fq, ft, 0.05, k=2)
    assert n == 2
    assert s > 1.0          # sum of two cosines over ONE target — like gpu_score


def _pf(pts, feats):
    return PointFeatures(pts=np.asarray(pts, float),
                         feats=np.asarray(feats, np.float32))


_PTS_Q = [[0.0, 0, 0]]
_PTS_T = [[0.0, 0, 0], [1.0, 0, 0]]
_FQ = [[1.0, 0]]
_FT = [[1.0, 0], [1.0, 0]]


def test_champion_scorer_eq5_mode_uses_eq5_for_both_terms():
    q, t = _pf(_PTS_Q, _FQ), _pf(_PTS_T, _FT)
    pose = PoseHypothesis(I3, Z3, 1.0,
                          {"s_icp": 0.5, "R_coarse": I3.copy(),
                           "t_coarse": Z3.copy()})
    sc = ChampionScorer(use_s_coarse=True, eq5_terms=True, tau_abs=0.05)
    out = sc.score(pose, q, t)
    ref, _ = eq5_score(I3, Z3, np.asarray(_PTS_Q), np.asarray(_PTS_T),
                       np.asarray(_FQ, np.float32), np.asarray(_FT, np.float32),
                       0.05)
    assert ref == pytest.approx(0.5)
    assert out.breakdown["s_feat_1"] == pytest.approx(ref)
    assert out.breakdown["s_coarse"] == pytest.approx(ref)
    assert out.score == pytest.approx(0.5 * ref * ref)   # s_icp * s1 * sc


def test_champion_scorer_default_stays_mean_cosine():
    q, t = _pf(_PTS_Q, _FQ), _pf(_PTS_T, _FT)
    pose = PoseHypothesis(I3, Z3, 1.0, {"s_icp": 0.5})
    out = ChampionScorer(tau_abs=0.05).score(pose, q, t)
    # Geometric NN: only target 0 is an inlier, mean cosine = 1.0 — the
    # historical value, NOT Eq.5's 0.5.
    assert out.breakdown["s_feat_1"] == pytest.approx(1.0)


def test_stages_for_object_plumbs_eq5_terms():
    from popoe.freeze.recipes import stages_for_object
    _, _, scorer = stages_for_object(0.1, eq5_terms=True)
    assert scorer.eq5_terms is True
    _, _, scorer = stages_for_object(0.1)
    assert scorer.eq5_terms is False
