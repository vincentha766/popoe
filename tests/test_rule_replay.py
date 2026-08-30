"""rule_replay over a cand dump (csv module)."""
import csv
import importlib.util
from pathlib import Path

import pytest

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "rule_replay.py"


@pytest.fixture(scope="module")
def rr():
    spec = importlib.util.spec_from_file_location("rule_replay", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows():
    return [
        dict(scene_id=1, im_id=1, obj_id=5, cand=0, w=1.0,
             s_icp=0.9, s_feat_1=0.2, metric_fit=1.0, s_coarse=0.1, score=0.18,
             R="1 0 0 0 1 0 0 0 1", t="0 0 0"),
        dict(scene_id=1, im_id=1, obj_id=5, cand=1, w=1.0,
             s_icp=0.3, s_feat_1=0.9, metric_fit=1.0, s_coarse=0.8, score=0.27,
             R="1 0 0 0 1 0 0 0 1", t="1 1 1"),
        dict(scene_id=1, im_id=2, obj_id=5, cand=0, w=1.0,
             s_icp=0.5, s_feat_1=0.5, metric_fit=1.0, s_coarse=0.5, score=0.25,
             R="1 0 0 0 1 0 0 0 1", t="2 2 2"),
    ]


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_parse_rule_terms_and_exponents(rr):
    cols = _rows()[0].keys()
    assert rr.parse_rule("s_icp * s_feat_1", cols) == {"s_icp": 1.0, "s_feat_1": 1.0}
    assert rr.parse_rule("s_feat_1^0.5", cols) == {"s_feat_1": 0.5}
    assert rr.parse_rule("s_icp*s_icp", cols) == {"s_icp": 2.0}


def test_parse_rule_missing_column_is_loud(rr):
    cols = [c for c in _rows()[0] if c != "s_coarse"]
    with pytest.raises(SystemExit, match="score-coarse"):
        rr.parse_rule("s_icp * s_coarse", cols)
    with pytest.raises(SystemExit, match="not in the dump"):
        rr.parse_rule("s_icp * nonsense", cols)


def test_parse_rule_rejects_non_numeric_columns(rr):
    cols = list(_rows()[0].keys()) + ["solver", "source", "R_prererank",
                                      "t_prererank"]
    for bad in ("solver", "source", "R", "R_prererank", "t_prererank",
                "scene_id"):
        with pytest.raises(SystemExit, match="not a numeric scoring term"):
            rr.parse_rule(f"s_icp * {bad}", cols)


def test_parse_rule_rejects_negative_exponent(rr):
    with pytest.raises(SystemExit, match="negative exponent"):
        rr.parse_rule("s_feat_1^-1", _rows()[0].keys())


def test_flip_counted_for_same_cand_w_different_hypothesis(rr):
    rows = [
        dict(scene_id=1, im_id=1, obj_id=5, cand=0, w=1.0,
             s_icp=0.9, s_feat_1=0.2, score=0.5, R="1 0 0 0 1 0 0 0 1", t="0 0 0"),
        dict(scene_id=1, im_id=1, obj_id=5, cand=0, w=1.0,
             s_icp=0.3, s_feat_1=0.9, score=0.1, R="0 1 0 1 0 0 0 0 1", t="9 9 9"),
    ]
    base = rr.champion_index(rows, [float(r["score"]) for r in rows])
    rule = rr.champion_index(rows, rr.rule_score(rows, {"s_feat_1": 1.0}))
    assert sum(1 for k in rule if rule[k] != base[k]) == 1


def test_rule_score_clamps_negative_and_applies_exponents(rr):
    rows = [{"s_icp": 0.5, "s_feat_1": 0.4}, {"s_icp": -0.2, "s_feat_1": 0.9}]
    s = rr.rule_score(rows, {"s_icp": 1.0, "s_feat_1": 2.0})
    assert s[0] == pytest.approx(0.5 * 0.4 ** 2)
    assert s[1] == 0.0


def test_champions_pick_per_target_argmax(rr):
    rows = _rows()
    champs = rr.champions(rows, rr.rule_score(rows, {"s_icp": 1.0}))
    a = [c for c in champs if int(c["im_id"]) == 1]
    assert len(a) == 1 and int(a[0]["cand"]) == 0
    champs2 = rr.champions(rows, rr.rule_score(rows, {"s_feat_1": 1.0}))
    a2 = [c for c in champs2 if int(c["im_id"]) == 1]
    assert int(a2[0]["cand"]) == 1
    assert len(champs) == len(champs2) == 2


def test_complete_results_zero_pads_and_preserves_target_order(rr):
    rows = _rows()
    champs = rr.champions(rows, [float(r["score"]) for r in rows])
    targets = [
        {"scene_id": 1, "im_id": 2, "obj_id": 5},
        {"scene_id": 1, "im_id": 3, "obj_id": 5},
        {"scene_id": 1, "im_id": 1, "obj_id": 5},
    ]
    out, n_missed = rr.complete_results(champs, targets)
    assert n_missed == 1
    assert [int(r["im_id"]) for r in out] == [2, 3, 1]
    missed = next(r for r in out if int(r["im_id"]) == 3)
    assert float(missed["score"]) == 0.0
    assert missed["R"] == rr.ZERO_R
    assert missed["t"] == rr.ZERO_T


def test_complete_results_rejects_target_outside_universe(rr):
    targets = [{"scene_id": 1, "im_id": 1, "obj_id": 5}]
    champs = rr.champions(_rows(), [float(r["score"]) for r in _rows()])
    with pytest.raises(SystemExit, match="outside --target-csv"):
        rr.complete_results(champs, targets)


def test_load_target_universe_rejects_duplicate_keys(rr, tmp_path):
    path = tmp_path / "duplicate_targets.csv"
    _write(path, [
        {"scene_id": 1, "im_id": 1, "obj_id": 5},
        {"scene_id": 1, "im_id": 1, "obj_id": 5},
    ])
    with pytest.raises(SystemExit, match="duplicate target keys"):
        rr.load_target_universe(path)


def _run_cli(rr, argv):
    import sys
    old = sys.argv
    try:
        sys.argv = ["rule_replay.py"] + argv
        rr.main()
    finally:
        sys.argv = old


def test_warns_on_mixed_solver_dump(rr, tmp_path, capsys):
    rows = _rows()
    for r, s in zip(rows, ["o3d", "gpu", "o3d"]):
        r["solver"] = s
    csv_path = tmp_path / "mixed.csv"
    _write(csv_path, rows)
    _run_cli(rr, [str(csv_path), "--rule", "s_icp"])
    assert "mixes 2 solvers" in capsys.readouterr().out


def test_end_to_end_writes_results_and_reports_flips(rr, tmp_path, capsys):
    csv_path = tmp_path / "cands.csv"
    _write(csv_path, _rows())
    _run_cli(rr, [str(csv_path), "--rule", "s_icp", "--rule", "s_feat_1",
                  "--out-dir", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "2 targets" in out
    assert "1/2 targets flip" in out and "0/2 targets flip" in out
    res = _read(tmp_path / "out" / "replay_s_icp.csv")
    assert list(res[0].keys()) == ["scene_id", "im_id", "obj_id", "score", "R", "t"]
    assert len(res) == 2


def _pool_rows():
    return [
        dict(scene_id=1, im_id=1, obj_id=5, cand=0, w=1.0, source="cnos",
             s_icp=0.9, s_feat_1=0.2, metric_fit=1.0, score=0.18,
             R="1 0 0 0 1 0 0 0 1", t="0 0 0"),
        dict(scene_id=1, im_id=1, obj_id=5, cand=1, w=0.7, source="cnos",
             s_icp=0.3, s_feat_1=0.9, metric_fit=1.0, score=0.27,
             R="1 0 0 0 1 0 0 0 1", t="1 1 1"),
        dict(scene_id=1, im_id=1, obj_id=5, cand=2, w=0.7, source="muse",
             s_icp=0.8, s_feat_1=0.8, metric_fit=1.0, score=0.64,
             R="1 0 0 0 1 0 0 0 1", t="2 2 2"),
        dict(scene_id=1, im_id=2, obj_id=5, cand=0, w=0.7, source="muse",
             s_icp=0.5, s_feat_1=0.5, metric_fit=1.0, score=0.25,
             R="1 0 0 0 1 0 0 0 1", t="3 3 3"),
    ]


def test_restrict_keeps_only_requested_values(rr):
    rows = _pool_rows()
    kept, label = rr.restrict(rows, "w", [0.7], rr._round6)
    assert len(kept) == 3 and {rr._round6(r["w"]) for r in kept} == {0.7}
    assert label == "w=0.7"
    kept, label = rr.restrict(rows, "source", ["cnos"], str)
    assert {r["source"] for r in kept} == {"cnos"} and label == "source=cnos"


def test_restrict_unknown_value_is_loud(rr):
    rows = _pool_rows()
    with pytest.raises(SystemExit, match="absent from the dump"):
        rr.restrict(rows, "source", ["nids"], str)
    with pytest.raises(SystemExit, match="absent from the dump"):
        rr.restrict(rows, "w", [0.3], rr._round6)


def test_restrict_missing_column_is_loud(rr):
    rows = [{k: v for k, v in r.items() if k != "source"} for r in _pool_rows()]
    with pytest.raises(SystemExit, match="no 'source' column"):
        rr.restrict(rows, "source", ["cnos"], str)


def test_keep_source_arm_zero_pads_targets_it_empties(rr, tmp_path, capsys):
    csv_path = tmp_path / "cands.csv"
    _write(csv_path, _pool_rows())
    targets = tmp_path / "poses.csv"
    _write(targets, [{c: r[c] for c in rr.KEY} for r in
                     {(r["scene_id"], r["im_id"], r["obj_id"]): r
                      for r in _pool_rows()}.values()])
    _run_cli(rr, [str(csv_path), "--rule", "s_icp*s_feat_1*metric_fit",
                  "--target-csv", str(targets), "--keep-source", "cnos",
                  "--out-dir", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "kept 2/4 hypotheses, 1/2 candidate-bearing targets" in out
    assert "2 targets (1 zero-padded)" in out
    res = _read(tmp_path / "out" /
                "replay_s_icp_s_feat_1_metric_fit_source_cnos.csv")
    assert len(res) == 2
    assert float(next(r for r in res if r["im_id"] == "2")["score"]) == 0.0


def test_keep_obj_slices_to_the_ablation_objects(rr, tmp_path, capsys):
    rows = _pool_rows()
    for r in rows:
        if int(r["im_id"]) == 2:
            r["obj_id"] = 19
    csv_path = tmp_path / "cands.csv"
    _write(csv_path, rows)
    targets = tmp_path / "poses.csv"
    sliced = [r for r in rows if int(r["obj_id"]) == 19]
    _write(targets, [{c: r[c] for c in rr.KEY} for r in
                     {(r["scene_id"], r["im_id"], r["obj_id"]): r
                      for r in sliced}.values()])
    _run_cli(rr, [str(csv_path), "--rule", "s_icp*s_feat_1*metric_fit",
                  "--target-csv", str(targets), "--keep-obj", "19",
                  "--out-dir", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "kept 1/4 hypotheses" in out
    res = _read(tmp_path / "out" /
                "replay_s_icp_s_feat_1_metric_fit_obj_id_19.csv")
    assert len(res) == 1 and {int(r["obj_id"]) for r in res} == {19}


def test_keep_w_arm_restricts_before_argmax(rr, tmp_path, capsys):
    csv_path = tmp_path / "cands.csv"
    _write(csv_path, _pool_rows())
    _run_cli(rr, [str(csv_path), "--rule", "s_icp*s_feat_1*metric_fit",
                  "--keep-w", "1.0", "--out-dir", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "kept 1/4 hypotheses" in out
    assert "1 lost every candidate" in out
    assert "NOT comparable" in out
    res = _read(tmp_path / "out" /
                "replay_s_icp_s_feat_1_metric_fit_w_1_0.csv")
    assert len(res) == 1 and res[0]["t"] == "0 0 0"


def test_end_to_end_target_csv_zero_pads_missing_targets(rr, tmp_path, capsys):
    csv_path = tmp_path / "cands.csv"
    _write(csv_path, _rows())
    targets = tmp_path / "poses.csv"
    _write(targets, [
        {"scene_id": 1, "im_id": 1, "obj_id": 5},
        {"scene_id": 1, "im_id": 2, "obj_id": 5},
        {"scene_id": 1, "im_id": 3, "obj_id": 5},
    ])
    _run_cli(rr, [str(csv_path), "--rule", "s_icp",
                  "--target-csv", str(targets), "--out-dir", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "3 targets (1 zero-padded)" in out
    res = _read(tmp_path / "out" / "replay_s_icp.csv")
    assert len(res) == 3
    missed = next(r for r in res if r["im_id"] == "3")
    assert float(missed["score"]) == 0.0
    assert missed["R"] == rr.ZERO_R
    assert missed["t"] == rr.ZERO_T
