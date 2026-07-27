"""bop_time_normalize: the BOP `time` rule (one shared value per image, or
uniform -1) derived from bop_eval's raw per-target column.

The trap the per-TARGET sum exists for: a multi-instance target writes
inst_count rows that all carry the target's ONE elapsed, so a per-row sum
would double-count exactly on the datasets (tless/icbin/itodd) this tool is
needed for.

Loads the example module by path (examples/ is not a package), like
test_bop_eval_cli.py.
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_EXAMPLE = (Path(__file__).resolve().parents[1] / "examples"
            / "bop_time_normalize.py")


@pytest.fixture(scope="module")
def tn():
    spec = importlib.util.spec_from_file_location("bop_time_normalize",
                                                  _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(scene, im, obj, t):
    return {"scene_id": str(scene), "im_id": str(im), "obj_id": str(obj),
            "score": "0.9", "R": "1 0 0 0 1 0 0 0 1", "t": "0 0 0",
            "time": t}


# one image: obj 7 has inst_count 2 (two rows, one elapsed), obj 9 has 1
_ROWS = [_row(1, 3, 7, "5.000"), _row(1, 3, 7, "5.000"),
         _row(1, 3, 9, "2.000")]


def test_multi_instance_rows_are_not_double_counted(tn):
    out, uncovered = tn.normalize([dict(r) for r in _ROWS], [], "sum")
    assert [r["time"] for r in out] == ["7.000"] * 3   # 5+2, not 5+5+2
    assert uncovered == 0


def test_detection_time_is_added_per_image(tn):
    dmap = {(1, 3): 1.5}
    out, uncovered = tn.normalize([dict(r) for r in _ROWS], [dmap], "sum")
    assert [r["time"] for r in out] == ["8.500"] * 3
    assert uncovered == 0


def test_image_absent_from_detections_is_counted_not_silent(tn):
    out, uncovered = tn.normalize([dict(r) for r in _ROWS], [{(9, 9): 1.0}],
                                  "sum")
    assert [r["time"] for r in out] == ["7.000"] * 3
    assert uncovered == 1


def test_neg1_mode_is_uniform(tn):
    rows = [dict(r) for r in _ROWS] + [_row(2, 1, 5, "")]  # legacy row too
    out, _ = tn.normalize(rows, [], "neg1")
    assert all(r["time"] == "-1" for r in out)


def test_disagreeing_target_rows_are_fatal(tn):
    rows = [_row(1, 3, 7, "5.000"), _row(1, 3, 7, "6.000")]
    with pytest.raises(SystemExit, match="two different time"):
        tn.normalize(rows, [], "sum")


def test_legacy_empty_time_is_fatal_in_sum_mode(tn):
    with pytest.raises(SystemExit, match="neg1"):
        tn.normalize([_row(1, 3, 7, "")], [], "sum")


def test_detections_without_time_field_are_fatal(tn, tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"scene_id": 1, "image_id": 3,
                              "category_id": 7, "score": 0.9}]))
    with pytest.raises(SystemExit, match="no 'time' field"):
        tn.detection_times(str(p))


def test_detection_times_take_the_per_image_max(tn, tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([
        {"scene_id": 1, "image_id": 3, "time": 1.2},
        {"scene_id": 1, "image_id": 3, "time": 1.3},
        {"scene_id": 1, "image_id": 4, "time": 0.7}]))
    assert tn.detection_times(str(p)) == {(1, 3): 1.3, (1, 4): 0.7}


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["scene_id", "im_id", "obj_id",
                                           "score", "R", "t", "time"])
        wr.writeheader()
        wr.writerows(rows)


def test_cli_round_trip(tn, tmp_path, monkeypatch):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "sub.csv"
    dets = tmp_path / "dets.json"
    _write_csv(raw, _ROWS)
    dets.write_text(json.dumps([{"scene_id": 1, "image_id": 3, "time": 1.0}]))
    monkeypatch.setattr(sys, "argv", ["bop_time_normalize", str(raw),
                                      "--out", str(out),
                                      "--detections", str(dets)])
    tn.main()
    got = list(csv.DictReader(open(out)))
    assert [r["time"] for r in got] == ["8.000"] * 3
    # every row of one (scene, im) shares one value — the rule itself
    assert len({r["time"] for r in got}) == 1


def test_cli_refuses_in_place_overwrite(tn, tmp_path, monkeypatch):
    raw = tmp_path / "raw.csv"
    _write_csv(raw, _ROWS)
    monkeypatch.setattr(sys, "argv", ["bop_time_normalize", str(raw),
                                      "--out", str(raw)])
    with pytest.raises(SystemExit, match="differ"):
        tn.main()


def test_cli_rejects_a_non_pose_csv(tn, tmp_path, monkeypatch):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    monkeypatch.setattr(sys, "argv", ["bop_time_normalize", str(bad),
                                      "--out", str(tmp_path / "o.csv")])
    with pytest.raises(SystemExit, match="missing column"):
        tn.main()
