"""metrics.grasp — grasp-axis ADD(-S) port of gedi freezev2_grasp_eval.py.

The defining property here is the ARCHIVED calibre (deliberately NOT flat):
mean over per-object recalls, median over per-object medians. The archived
REPRODUCTION.md grasp rows (#5/#6) were produced at this calibre, so the port
must keep it for numbers to reconcile.
"""
import numpy as np
import pytest

from popoe.metrics import grasp


def test_aggregate_grasp_is_per_object_calibre():
    # obj 1: 1 instance, always wrong; obj 2: 9 instances, always right.
    # Per-object mean recall = (0 + 1) / 2 = 0.5; flat would say 0.9.
    obj = {
        1: dict(pts=None, diam=100.0, sym=False),
        2: dict(pts=None, diam=100.0, sym=True),
    }
    err = {
        1: [(1e9, 50.0, 90.0)],
        2: [(1.0, 2.0, 3.0)] * 9,
    }
    per_object, summary = grasp.aggregate_grasp(err, obj)
    assert summary["add_s_recall_01d"] == pytest.approx(0.5)
    assert summary["add_s_recall_01d"] != pytest.approx(0.9)
    # Median of per-object medians, not of the pooled instances.
    assert summary["median_t_mm"] == pytest.approx(np.median([50.0, 2.0]))
    assert per_object[2]["sym"] is True


def test_recall_thresholds_scale_with_diameter():
    obj = {7: dict(pts=None, diam=200.0, sym=False)}
    # 0.1d = 20.0: errors 19 passes, 21 fails; 0.05d = 10.0: 9 passes only.
    err = {7: [(19.0, 1.0, 1.0), (21.0, 1.0, 1.0), (9.0, 1.0, 1.0)]}
    _, summary = grasp.aggregate_grasp(err, obj)
    assert summary["add_s_recall_01d"] == pytest.approx(2 / 3)
    assert summary["add_s_recall_005d"] == pytest.approx(1 / 3)


def test_per_object_errors_picks_best_gt_and_uses_sym(monkeypatch):
    calls = []

    class FakePoseError:
        @staticmethod
        def add(Re, te, Rg, tg, pts):
            calls.append("add")
            return float(np.linalg.norm(te - tg))

        @staticmethod
        def adi(Re, te, Rg, tg, pts):
            calls.append("adi")
            return float(np.linalg.norm(te - tg))

        @staticmethod
        def te(te_, tg):
            return float(np.linalg.norm(te_ - tg))

        @staticmethod
        def re(Re, Rg):
            return 0.0

    monkeypatch.setattr(grasp, "_pose_error", lambda: FakePoseError)

    I9 = " ".join(["1", "0", "0", "0", "1", "0", "0", "0", "1"])
    rows = [dict(scene_id="1", im_id="1", obj_id="5", R=I9, t="0 0 100")]
    # Two GT instances; the closer one (t=[0,0,110], err 10) must win over 90.
    gt = {(1, 1, 5): [
        dict(R=np.eye(3), t=np.array([[0.0], [0.0], [190.0]])),
        dict(R=np.eye(3), t=np.array([[0.0], [0.0], [110.0]])),
    ]}
    obj = {5: dict(pts=np.zeros((3, 3)), diam=100.0, sym=True)}

    err = grasp.per_object_errors(rows, gt, obj)
    assert set(calls) == {"adi"}  # symmetric object -> ADD-S path
    assert err[5][0][0] == pytest.approx(10.0)  # best GT, not first GT


def test_rows_without_gt_are_skipped():
    monkey_rows = [dict(scene_id="9", im_id="9", obj_id="9", R="0" * 1, t="0")]
    # No GT for the key -> no crash, no entry (matches archived behaviour).
    class _NoTouch:
        pass
    err = grasp.per_object_errors(monkey_rows, gt={}, obj={})
    assert err == {}
