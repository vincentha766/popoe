"""Unit tests for dual-CAD confusable assignment (CPU only)."""

from popoe.confusable_select import (
    dual_assign_hyps,
    dual_assign_rows,
    partner_id,
    product_score,
    score_from_row,
)
from popoe.interfaces import PoseHypothesis
from popoe.freeze.recipes import YCBV_MERGE_LABELS
import numpy as np


def test_product_score_clamps_negative():
    assert product_score(-1, 0.5, 1.0) == 0.0
    assert product_score(0.5, 0.5, 0.5) == 0.125


def test_dual_assign_gives_mask_to_higher_metric_fit():
    # cand0: better metric_fit for obj20; cand1: better for obj19
    q19 = [
        {"cand": 0, "w": 1.0, "s_icp": 0.4, "s_feat_1": 0.5, "metric_fit": 0.2},
        {"cand": 1, "w": 1.0, "s_icp": 0.3, "s_feat_1": 0.5, "metric_fit": 0.8},
    ]
    q20 = [
        {"cand": 0, "w": 1.0, "s_icp": 0.4, "s_feat_1": 0.5, "metric_fit": 0.9},
        {"cand": 1, "w": 1.0, "s_icp": 0.3, "s_feat_1": 0.5, "metric_fit": 0.1},
    ]
    p19 = dual_assign_rows(q19, q20)
    p20 = dual_assign_rows(q20, q19)
    assert p19.cand == 1
    assert p20.cand == 0
    assert p19.fallback is False


def test_dual_assign_fallback_without_partner():
    rows = [
        {"cand": 0, "s_icp": 0.1, "s_feat_1": 0.5, "metric_fit": 0.5},
        {"cand": 1, "s_icp": 0.4, "s_feat_1": 0.5, "metric_fit": 0.5},
    ]
    pick = dual_assign_rows(rows, [])
    assert pick.fallback is True
    assert pick.cand == 1
    assert score_from_row(pick.row_or_hyp) == product_score(0.4, 0.5, 0.5)


def test_dual_assign_hyps_matches_rows():
    R = np.eye(3)
    t = np.zeros(3)

    def H(s_icp, s_feat, mf, score=None):
        sc = score if score is not None else product_score(s_icp, s_feat, mf)
        return PoseHypothesis(
            R=R, t=t, score=sc,
            breakdown={"s_icp": s_icp, "s_feat_1": s_feat, "metric_fit": mf},
        )

    q = {0: [H(0.4, 0.5, 0.2)], 1: [H(0.3, 0.5, 0.8)]}
    p = {0: [H(0.4, 0.5, 0.9)], 1: [H(0.3, 0.5, 0.1)]}
    best = dual_assign_hyps(q, p)
    assert best.breakdown["metric_fit"] == 0.8  # cand 1 for query


def test_partner_id_ycbv():
    assert partner_id(19, YCBV_MERGE_LABELS) == 20
    assert partner_id(20, YCBV_MERGE_LABELS) == 19
    assert partner_id(5, YCBV_MERGE_LABELS) is None
