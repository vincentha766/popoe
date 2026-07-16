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
