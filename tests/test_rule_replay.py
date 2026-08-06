"""rule_replay: replay arbitration rules over a cand dump (pure pandas). Tests
the core (parse_rule / rule_score / champions) + the loud missing-column guard.
"""
import importlib.util
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "rule_replay.py"


@pytest.fixture(scope="module")
def rr():
    spec = importlib.util.spec_from_file_location("rule_replay", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _df():
    # two targets; each with candidate rows differing in which term wins.
    rows = [
        # target A: cand 0 has high s_icp, cand 1 has high s_feat_1
        dict(scene_id=1, im_id=1, obj_id=5, cand=0, w=1.0,
             s_icp=0.9, s_feat_1=0.2, metric_fit=1.0, s_coarse=0.1, score=0.18,
             R="1 0 0 0 1 0 0 0 1", t="0 0 0"),
        dict(scene_id=1, im_id=1, obj_id=5, cand=1, w=1.0,
             s_icp=0.3, s_feat_1=0.9, metric_fit=1.0, s_coarse=0.8, score=0.27,
             R="1 0 0 0 1 0 0 0 1", t="1 1 1"),
        # target B: single candidate
        dict(scene_id=1, im_id=2, obj_id=5, cand=0, w=1.0,
             s_icp=0.5, s_feat_1=0.5, metric_fit=1.0, s_coarse=0.5, score=0.25,
             R="1 0 0 0 1 0 0 0 1", t="2 2 2"),
    ]
    return pd.DataFrame(rows)


def test_parse_rule_terms_and_exponents(rr):
    df = _df()
    assert rr.parse_rule("s_icp * s_feat_1", df.columns) == {"s_icp": 1.0, "s_feat_1": 1.0}
    assert rr.parse_rule("s_feat_1^0.5", df.columns) == {"s_feat_1": 0.5}
    assert rr.parse_rule("s_icp*s_icp", df.columns) == {"s_icp": 2.0}  # accumulates


def test_parse_rule_missing_column_is_loud(rr):
    df = _df().drop(columns=["s_coarse"])
    with pytest.raises(SystemExit, match="score-coarse"):
        rr.parse_rule("s_icp * s_coarse", df.columns)
    with pytest.raises(SystemExit, match="not in the dump"):
        rr.parse_rule("s_icp * nonsense", df.columns)


def test_parse_rule_rejects_non_numeric_columns(rr):
    """id / pose / provenance columns (incl. the new `solver`) are not scoring
    terms — referencing one is a loud error, not a string-arithmetic crash."""
    cols = list(_df().columns) + ["solver", "source", "R_prererank",
                                  "t_prererank"]
    for bad in ("solver", "source", "R", "R_prererank", "t_prererank",
                "scene_id"):
        with pytest.raises(SystemExit, match="not a numeric scoring term"):
            rr.parse_rule(f"s_icp * {bad}", cols)


def test_parse_rule_rejects_negative_exponent(rr):
    df = _df()
    with pytest.raises(SystemExit, match="negative exponent"):
        rr.parse_rule("s_feat_1^-1", df.columns)


def test_flip_counted_for_same_cand_w_different_hypothesis(rr):
    """Two hypotheses share (cand, w) but differ in pose (e.g. n_restarts>1).
    A flip must be detected by ROW identity, not (cand, w)."""
    df = pd.DataFrame([
        dict(scene_id=1, im_id=1, obj_id=5, cand=0, w=1.0,
             s_icp=0.9, s_feat_1=0.2, score=0.5, R="1 0 0 0 1 0 0 0 1", t="0 0 0"),
        dict(scene_id=1, im_id=1, obj_id=5, cand=0, w=1.0,           # same (cand,w)
             s_icp=0.3, s_feat_1=0.9, score=0.1, R="0 1 0 1 0 0 0 0 1", t="9 9 9"),
    ])
    base = rr.champion_index(df, df["score"])          # picks row 0 (score 0.5)
    rule = rr.champion_index(df, rr.rule_score(df, {"s_feat_1": 1.0}))  # row 1
    assert int((rule != base).sum()) == 1              # flip detected


def test_rule_score_clamps_negative_and_applies_exponents(rr):
    df = pd.DataFrame({"s_icp": [0.5, -0.2], "s_feat_1": [0.4, 0.9]})
    s = rr.rule_score(df, {"s_icp": 1.0, "s_feat_1": 2.0})
    assert s.iloc[0] == pytest.approx(0.5 * 0.4 ** 2)
    assert s.iloc[1] == 0.0                       # negative s_icp clamped to 0


def test_champions_pick_per_target_argmax(rr):
    df = _df()
    # rule = s_icp -> target A picks cand 0 (0.9 > 0.3)
    champs = rr.champions(df, rr.rule_score(df, {"s_icp": 1.0}))
    a = champs[(champs.scene_id == 1) & (champs.im_id == 1)]
    assert len(a) == 1 and a.iloc[0]["cand"] == 0
    # rule = s_feat_1 -> target A flips to cand 1 (0.9 > 0.2)
    champs2 = rr.champions(df, rr.rule_score(df, {"s_feat_1": 1.0}))
    a2 = champs2[(champs2.scene_id == 1) & (champs2.im_id == 1)]
    assert a2.iloc[0]["cand"] == 1
    # exactly one champion per target either way
    assert len(champs) == len(champs2) == 2


def test_complete_results_zero_pads_and_preserves_target_order(rr):
    champs = rr.champions(_df(), _df()["score"])
    targets = pd.DataFrame([
        {"scene_id": 1, "im_id": 2, "obj_id": 5},
        {"scene_id": 1, "im_id": 3, "obj_id": 5},
        {"scene_id": 1, "im_id": 1, "obj_id": 5},
    ])
    out, n_missed = rr.complete_results(champs, targets)
    assert n_missed == 1
    assert list(out["im_id"]) == [2, 3, 1]
    missed = out[out["im_id"] == 3].iloc[0]
    assert missed["score"] == 0.0
    assert missed["R"] == rr.ZERO_R
    assert missed["t"] == rr.ZERO_T


def test_complete_results_rejects_target_outside_universe(rr):
    targets = pd.DataFrame([
        {"scene_id": 1, "im_id": 1, "obj_id": 5},
    ])
    champs = rr.champions(_df(), _df()["score"])
    with pytest.raises(SystemExit, match="outside --target-csv"):
        rr.complete_results(champs, targets)


def test_load_target_universe_rejects_duplicate_keys(rr, tmp_path):
    targets = pd.DataFrame([
        {"scene_id": 1, "im_id": 1, "obj_id": 5},
        {"scene_id": 1, "im_id": 1, "obj_id": 5},
    ])
    path = tmp_path / "duplicate_targets.csv"
    targets.to_csv(path, index=False)
    with pytest.raises(SystemExit, match="duplicate target keys"):
        rr.load_target_universe(path)


def test_warns_on_mixed_solver_dump(rr, tmp_path, capsys):
    df = _df()
    df["solver"] = ["o3d", "gpu", "o3d"]           # two solvers in one dump
    csv = tmp_path / "mixed.csv"
    df.to_csv(csv, index=False)
    import sys
    old = sys.argv
    try:
        sys.argv = ["rule_replay.py", str(csv), "--rule", "s_icp"]
        rr.main()
    finally:
        sys.argv = old
    assert "mixes 2 solvers" in capsys.readouterr().out


def test_end_to_end_writes_results_and_reports_flips(rr, tmp_path, capsys):
    csv = tmp_path / "cands.csv"
    _df().to_csv(csv, index=False)
    import sys
    argv = [str(csv), "--rule", "s_icp", "--rule", "s_feat_1",
            "--out-dir", str(tmp_path / "out")]
    old = sys.argv
    try:
        sys.argv = ["rule_replay.py"] + argv
        rr.main()
    finally:
        sys.argv = old
    out = capsys.readouterr().out
    assert "2 targets" in out
    # baseline = dump 'score' col: target A champion is cand1 (0.27>0.18).
    # rule s_icp picks cand0 for A -> 1 flip; rule s_feat_1 picks cand1 -> 0 flips.
    assert "1/2 targets flip" in out and "0/2 targets flip" in out
    res = pd.read_csv(tmp_path / "out" / "replay_s_icp.csv")
    assert list(res.columns) == ["scene_id", "im_id", "obj_id", "score", "R", "t"]
    assert len(res) == 2                          # one row per target


def _pool_df():
    """Two targets. Target A has one candidate per (source, w); target B is
    MUSE-only at w=0.7 — so restricting the pool can empty a target entirely."""
    rows = [
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
    return pd.DataFrame(rows)


def test_restrict_keeps_only_requested_values(rr):
    df = _pool_df()
    kept, label = rr.restrict(df, "w", [0.7], rr._round6)
    assert len(kept) == 3 and set(kept["w"]) == {0.7}
    assert label == "w=0.7"
    kept, label = rr.restrict(df, "source", ["cnos"], str)
    assert set(kept["source"]) == {"cnos"} and label == "source=cnos"


def test_restrict_unknown_value_is_loud(rr):
    """A typo'd source must not read as 'this source contributes nothing'."""
    df = _pool_df()
    with pytest.raises(SystemExit, match="absent from the dump"):
        rr.restrict(df, "source", ["nids"], str)
    with pytest.raises(SystemExit, match="absent from the dump"):
        rr.restrict(df, "w", [0.3], rr._round6)


def test_restrict_missing_column_is_loud(rr):
    df = _pool_df().drop(columns=["source"])
    with pytest.raises(SystemExit, match="no 'source' column"):
        rr.restrict(df, "source", ["cnos"], str)


def _run_cli(rr, argv):
    import sys
    old = sys.argv
    try:
        sys.argv = ["rule_replay.py"] + argv
        rr.main()
    finally:
        sys.argv = old


def test_keep_source_arm_zero_pads_targets_it_empties(rr, tmp_path, capsys):
    """C4: restricting to CNOS leaves target B (MUSE-only) with no candidate.
    With --target-csv it stays in the denominator as a zero-score failure —
    otherwise the arm would score 1 target and look artificially strong."""
    csv = tmp_path / "cands.csv"
    _pool_df().to_csv(csv, index=False)
    targets = tmp_path / "poses.csv"
    _pool_df()[rr.KEY].drop_duplicates().to_csv(targets, index=False)
    _run_cli(rr, [str(csv), "--rule", "s_icp*s_feat_1*metric_fit",
                  "--target-csv", str(targets), "--keep-source", "cnos",
                  "--out-dir", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "kept 2/4 hypotheses, 1/2 candidate-bearing targets" in out
    assert "2 targets (1 zero-padded)" in out
    res = pd.read_csv(tmp_path / "out" /
                      "replay_s_icp_s_feat_1_metric_fit_source_cnos.csv")
    assert len(res) == 2
    assert res[res["im_id"] == 2].iloc[0]["score"] == 0.0


def test_keep_obj_slices_to_the_ablation_objects(rr, tmp_path, capsys):
    """C3 runs on YCB-V obj19/20 only, so the pooled arms must be sliced to the
    same objects the no-pool GPU arm produced — with a target universe sliced
    identically, or the replay trips the 'outside --target-csv' guard."""
    df = _pool_df()
    df.loc[df.im_id == 2, "obj_id"] = 19            # a second object in the dump
    csv = tmp_path / "cands.csv"
    df.to_csv(csv, index=False)
    targets = tmp_path / "poses.csv"
    df[df.obj_id == 19][rr.KEY].drop_duplicates().to_csv(targets, index=False)
    _run_cli(rr, [str(csv), "--rule", "s_icp*s_feat_1*metric_fit",
                  "--target-csv", str(targets), "--keep-obj", "19",
                  "--out-dir", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "kept 1/4 hypotheses" in out
    res = pd.read_csv(tmp_path / "out" /
                      "replay_s_icp_s_feat_1_metric_fit_obj_id_19.csv")
    assert len(res) == 1 and set(res["obj_id"]) == {19}


def test_keep_w_arm_restricts_before_argmax(rr, tmp_path, capsys):
    """C1: at w=1.0 only cand 0 survives for target A, so the champion is the
    one the adaptive arm did NOT pick (cand 2 at w=0.7 scores higher)."""
    csv = tmp_path / "cands.csv"
    _pool_df().to_csv(csv, index=False)
    _run_cli(rr, [str(csv), "--rule", "s_icp*s_feat_1*metric_fit",
                  "--keep-w", "1.0", "--out-dir", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "kept 1/4 hypotheses" in out
    assert "1 lost every candidate" in out
    assert "NOT comparable" in out          # no --target-csv -> loud warning
    res = pd.read_csv(tmp_path / "out" /
                      "replay_s_icp_s_feat_1_metric_fit_w_1_0.csv")
    assert len(res) == 1 and res.iloc[0]["t"] == "0 0 0"


def test_end_to_end_target_csv_zero_pads_missing_targets(rr, tmp_path, capsys):
    csv = tmp_path / "cands.csv"
    _df().to_csv(csv, index=False)
    targets = tmp_path / "poses.csv"
    pd.DataFrame([
        {"scene_id": 1, "im_id": 1, "obj_id": 5},
        {"scene_id": 1, "im_id": 2, "obj_id": 5},
        {"scene_id": 1, "im_id": 3, "obj_id": 5},
    ]).to_csv(targets, index=False)
    import sys
    old = sys.argv
    try:
        sys.argv = [
            "rule_replay.py", str(csv), "--rule", "s_icp",
            "--target-csv", str(targets), "--out-dir", str(tmp_path / "out"),
        ]
        rr.main()
    finally:
        sys.argv = old
    out = capsys.readouterr().out
    assert "3 targets (1 zero-padded)" in out
    res = pd.read_csv(tmp_path / "out" / "replay_s_icp.csv")
    assert len(res) == 3
    missed = res[res["im_id"] == 3].iloc[0]
    assert missed["score"] == 0.0
    assert missed["R"] == rr.ZERO_R
    assert missed["t"] == rr.ZERO_T
