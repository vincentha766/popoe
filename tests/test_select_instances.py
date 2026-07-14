"""adapters.select_top_instances — the BOP inst_count selection semantics.

Pure numpy; no GPU. The invariants that matter:
  * k=1 reproduces the old global argmax exactly (LMO/YCB-V unchanged);
  * k>1 returns champions of DISTINCT detections — never two hypotheses of the
    same detection, which would submit the same physical instance twice.
"""

import numpy as np

from popoe.adapters import BestScoreSelector, select_top_instances
from popoe.interfaces import PoseHypothesis


def _h(score, tag):
    return PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=score,
                          breakdown={"tag": tag})


SEL = BestScoreSelector()


def test_k1_equals_global_argmax():
    by_det = {0: [_h(0.3, "a"), _h(0.9, "b")],
              1: [_h(0.7, "c")],
              2: [_h(0.1, "d")]}
    got = select_top_instances(by_det, SEL, 1)
    assert [c.breakdown["tag"] for c in got] == ["b"]   # max over all hyps


def test_k2_takes_champions_of_distinct_detections():
    # Detection 0 holds the two highest-scoring hypotheses overall; a naive
    # global top-2 would return both ("a1", "a2") — the same instance twice.
    by_det = {0: [_h(0.95, "a1"), _h(0.90, "a2")],
              1: [_h(0.60, "b")],
              2: [_h(0.70, "c")]}
    got = select_top_instances(by_det, SEL, 2)
    assert [c.breakdown["tag"] for c in got] == ["a1", "c"]


def test_fewer_detections_than_k_returns_what_exists():
    by_det = {0: [_h(0.5, "a")]}
    got = select_top_instances(by_det, SEL, 3)
    assert len(got) == 1 and got[0].breakdown["tag"] == "a"


def test_empty_inputs():
    assert select_top_instances({}, SEL, 2) == []
    # a detection whose hypotheses all failed contributes nothing
    assert select_top_instances({0: []}, SEL, 2) == []


def test_champions_sorted_best_first():
    by_det = {0: [_h(0.2, "a")], 1: [_h(0.8, "b")], 2: [_h(0.5, "c")]}
    got = select_top_instances(by_det, SEL, 3)
    assert [c.breakdown["tag"] for c in got] == ["b", "c", "a"]
