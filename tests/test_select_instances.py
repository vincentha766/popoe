"""adapters.select_top_instances — the BOP inst_count selection semantics.

Pure numpy; no GPU. The invariants that matter:
  * k=1 reproduces the old global argmax exactly (LMO/YCB-V unchanged);
  * k>1 returns champions of DISTINCT detections — never two hypotheses of the
    same detection, which would submit the same physical instance twice.
"""

import numpy as np

from popoe.adapters import select_top_instances
from popoe.interfaces import PoseHypothesis


def _h(score, tag):
    return PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=score,
                          breakdown={"tag": tag})


def test_k1_equals_global_argmax():
    by_det = {0: [_h(0.3, "a"), _h(0.9, "b")],
              1: [_h(0.7, "c")],
              2: [_h(0.1, "d")]}
    got = select_top_instances(by_det, 1)
    assert [c.breakdown["tag"] for c in got] == ["b"]   # max over all hyps


def test_k2_takes_champions_of_distinct_detections():
    # Detection 0 holds the two highest-scoring hypotheses overall; a naive
    # global top-2 would return both ("a1", "a2") — the same instance twice.
    by_det = {0: [_h(0.95, "a1"), _h(0.90, "a2")],
              1: [_h(0.60, "b")],
              2: [_h(0.70, "c")]}
    got = select_top_instances(by_det, 2)
    assert [c.breakdown["tag"] for c in got] == ["a1", "c"]


def test_fewer_detections_than_k_returns_what_exists():
    by_det = {0: [_h(0.5, "a")]}
    got = select_top_instances(by_det, 3)
    assert len(got) == 1 and got[0].breakdown["tag"] == "a"


def test_empty_inputs():
    assert select_top_instances({}, 2) == []
    # a detection whose hypotheses all failed contributes nothing
    assert select_top_instances({0: []}, 2) == []


def test_champions_sorted_best_first():
    by_det = {0: [_h(0.2, "a")], 1: [_h(0.8, "b")], 2: [_h(0.5, "c")]}
    got = select_top_instances(by_det, 3)
    assert [c.breakdown["tag"] for c in got] == ["b", "c", "a"]


# ── §III-F translation NMS (nms_dist) ───────────────────────────────────

def _ht(score, tag, t):
    return PoseHypothesis(R=np.eye(3), t=np.asarray(t, float), score=score,
                          breakdown={"tag": tag})


def test_nms_suppressed_duplicate_frees_the_slot():
    # a and dup are one physical instance seen by two segmentors (2 mm apart);
    # c is a distinct instance 200 mm away. Without NMS the duplicate eats the
    # second slot; with it, the slot goes to c.
    by_det = {0: [_ht(0.90, "a", [0, 0, 0])],
              1: [_ht(0.85, "dup", [0.002, 0, 0])],
              2: [_ht(0.60, "c", [0.2, 0, 0])]}
    got = select_top_instances(by_det, 2, nms_dist=0.02)
    assert [c.breakdown["tag"] for c in got] == ["a", "c"]
    got = select_top_instances(by_det, 2)          # off by default
    assert [c.breakdown["tag"] for c in got] == ["a", "dup"]


def test_nms_does_not_pad_below_k():
    # Paper: retain only DISTINCT instance poses — no refilling with
    # suppressed duplicates when survivors fall short of k.
    by_det = {0: [_ht(0.9, "a", [0, 0, 0])],
              1: [_ht(0.8, "dup", [0.001, 0, 0])]}
    got = select_top_instances(by_det, 2, nms_dist=0.02)
    assert [c.breakdown["tag"] for c in got] == ["a"]


def test_nms_boundary_distance_is_kept():
    by_det = {0: [_ht(0.9, "a", [0, 0, 0])],
              1: [_ht(0.8, "b", [0.02, 0, 0])]}
    got = select_top_instances(by_det, 2, nms_dist=0.02)
    assert [c.breakdown["tag"] for c in got] == ["a", "b"]


def test_nms_keeps_the_higher_scored_of_a_duplicate_pair():
    by_det = {0: [_ht(0.7, "low", [0, 0, 0])],
              1: [_ht(0.9, "high", [0.001, 0, 0])]}
    got = select_top_instances(by_det, 1, nms_dist=0.02)
    assert [c.breakdown["tag"] for c in got] == ["high"]


def test_nms_is_greedy_against_kept_not_transitive():
    # b is inside a's radius (suppressed); c is inside b's radius but OUTSIDE
    # a's. Suppression compares against KEPT poses only, so c survives —
    # a suppressed duplicate must not veto its neighbours.
    by_det = {0: [_ht(0.9, "a", [0.0, 0, 0])],
              1: [_ht(0.8, "b", [0.015, 0, 0])],
              2: [_ht(0.7, "c", [0.03, 0, 0])]}
    got = select_top_instances(by_det, 3, nms_dist=0.02)
    assert [c.breakdown["tag"] for c in got] == ["a", "c"]


def test_nms_drops_nonfinite_translations_instead_of_letting_them_suppress():
    # NaN >= dist is False against every later champion, so one garbage row
    # would otherwise suppress the whole target. The garbage row is dropped;
    # with NMS off, legacy behaviour (garbage row kept) is unchanged.
    by_det = {0: [_ht(0.9, "bad", [float("nan"), 0, 0])],
              1: [_ht(0.8, "a", [1.0, 0, 0])],
              2: [_ht(0.7, "b", [2.0, 0, 0])]}
    got = select_top_instances(by_det, 3, nms_dist=0.02)
    assert [c.breakdown["tag"] for c in got] == ["a", "b"]
    got = select_top_instances(by_det, 3)
    assert [c.breakdown["tag"] for c in got] == ["bad", "a", "b"]
