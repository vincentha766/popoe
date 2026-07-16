"""S_coarse (pre-ICP feature score) as an opt-in diagnostic on ChampionScorer:
recorded in the breakdown WITHOUT changing the final score, so the evaluated
config stays byte-identical when off. numpy + sklearn (feature_aware_score);
ICPRefiner test needs open3d.
"""
import numpy as np
import pytest

# ChampionScorer.score() -> feature_aware_score -> pose_estimator, which hard-
# imports open3d at module load (reference extra). Skip cleanly without it.
pytest.importorskip("open3d")

from popoe.interfaces import PointFeatures, PoseHypothesis
from popoe.scoring import ChampionScorer


def _pf(pts, feats):
    return PointFeatures(pts=pts, feats=feats, meta={"feats_w1": feats})


def _identical_clouds(seed=0, n=60):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.05, 0.05, (n, 3))
    feats = rng.standard_normal((n, 8))
    return _pf(pts, feats), _pf(pts.copy(), feats.copy())


def test_s_coarse_off_is_byte_identical_and_absent():
    """compute_s_coarse=False: score is the plain s_icp*s_feat_1*met and the
    scorer adds NO s_coarse key."""
    q, t = _identical_clouds()
    pose = PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                          breakdown={"s_icp": 0.8})
    out = ChampionScorer(size_aware=False).score(pose, q, t)
    assert "s_coarse" not in out.breakdown
    assert out.score == pytest.approx(0.8 * out.breakdown["s_feat_1"])


def test_s_coarse_on_records_pre_icp_score_without_changing_score():
    """compute_s_coarse=True: s_coarse is feature_aware_score at the COARSE pose
    (breakdown R_coarse/t_coarse), distinct from post-ICP s_feat_1, and the
    final score is unchanged (diagnostic only)."""
    q, t = _identical_clouds()
    # refined pose = identity aligns the identical clouds perfectly (s_feat_1~1);
    # coarse pose = a 10 m translation -> zero inliers -> s_coarse = 0.
    pose = PoseHypothesis(
        R=np.eye(3), t=np.zeros(3), score=0.0,
        breakdown={"s_icp": 0.8, "R_coarse": np.eye(3),
                   "t_coarse": np.array([10.0, 10.0, 10.0])})
    on = ChampionScorer(size_aware=False, compute_s_coarse=True).score(pose, q, t)
    off = ChampionScorer(size_aware=False).score(
        PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                       breakdown={"s_icp": 0.8}), q, t)

    assert on.breakdown["s_feat_1"] == pytest.approx(1.0, abs=1e-6)  # aligned
    assert on.breakdown["s_coarse"] == 0.0                          # 10 m off
    assert on.breakdown["s_coarse"] != on.breakdown["s_feat_1"]
    # the final score is byte-identical to the s_coarse-off run
    assert on.score == pytest.approx(off.score)


def test_s_coarse_matches_s_feat_1_when_coarse_equals_refined():
    """Sanity: same pose for coarse and refined -> identical scores (proves
    s_coarse is the SAME formula, only the pose differs)."""
    q, t = _identical_clouds(seed=3)
    R = np.eye(3); tt = np.zeros(3)
    pose = PoseHypothesis(R=R, t=tt, score=0.0,
                          breakdown={"s_icp": 0.9, "R_coarse": R, "t_coarse": tt})
    out = ChampionScorer(size_aware=False, compute_s_coarse=True).score(pose, q, t)
    assert out.breakdown["s_coarse"] == pytest.approx(out.breakdown["s_feat_1"])


def test_s_coarse_uses_canonical_w1_features_not_reweighted():
    """s_coarse must be computed in the w=1 space (meta['feats_w1']), even when
    .feats are weight-scaled — the same canonical-space guarantee as s_feat_1."""
    rng = np.random.default_rng(7)
    pts = rng.uniform(-0.05, 0.05, (50, 3))
    w1 = rng.standard_normal((50, 8))            # canonical features (both sides)
    # .feats are DIFFERENT (as if reweighted / a wrong basis) on each side, so a
    # scorer that (wrongly) used .feats would get a low cosine, not ~1.
    q = PointFeatures(pts=pts, feats=rng.standard_normal((50, 8)),
                      meta={"feats_w1": w1})
    t = PointFeatures(pts=pts.copy(), feats=rng.standard_normal((50, 8)),
                      meta={"feats_w1": w1.copy()})
    pose = PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                          breakdown={"s_icp": 1.0, "R_coarse": np.eye(3),
                                     "t_coarse": np.zeros(3)})
    out = ChampionScorer(compute_s_coarse=True).score(pose, q, t)
    # identical w=1 feats + aligned pose -> cosine ~1; ~0 if it used .feats
    assert out.breakdown["s_coarse"] == pytest.approx(1.0, abs=1e-6)
    assert out.breakdown["s_feat_1"] == pytest.approx(1.0, abs=1e-6)


def test_s_coarse_off_preserves_existing_solver_s_coarse():
    """The solver stashes a (weight-scaled) breakdown['s_coarse']. With the flag
    OFF the scorer must NOT touch it; with ON it OVERWRITES with the canonical
    pre-ICP value."""
    q, t = _identical_clouds(seed=5)
    base = {"s_icp": 0.8, "s_coarse": 0.5}       # 0.5 = solver's weight-scaled value
    off = ChampionScorer().score(
        PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                       breakdown=dict(base)), q, t)
    assert off.breakdown["s_coarse"] == 0.5      # untouched

    on = ChampionScorer(compute_s_coarse=True).score(
        PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                       breakdown={**base, "R_coarse": np.eye(3),
                                  "t_coarse": np.zeros(3)}), q, t)
    assert on.breakdown["s_coarse"] == pytest.approx(1.0)  # canonical, overwritten
    assert on.breakdown["s_coarse"] != 0.5


def test_s_coarse_on_without_coarse_pose_raises():
    """A loud error, never a silent skip, if the coarse pose was not stashed."""
    q, t = _identical_clouds()
    pose = PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                          breakdown={"s_icp": 0.8})
    with pytest.raises(ValueError, match="coarse pose"):
        ChampionScorer(compute_s_coarse=True).score(pose, q, t)


def test_stages_for_object_wires_score_coarse_flag():
    from popoe.recipes import stages_for_object
    _, refiner_off, scorer_off = stages_for_object(0.1)
    assert refiner_off.keep_coarse is False and scorer_off.compute_s_coarse is False
    assert scorer_off.use_s_coarse is False
    _, refiner_on, scorer_on = stages_for_object(0.1, score_coarse=True)
    assert refiner_on.keep_coarse is True and scorer_on.compute_s_coarse is True
    # use_s_coarse implies keep_coarse AND compute_s_coarse (s_coarse is both
    # computed and emitted), even if score_coarse itself is False
    _, refiner_u, scorer_u = stages_for_object(0.1, use_s_coarse=True)
    assert refiner_u.keep_coarse is True and scorer_u.use_s_coarse is True
    assert scorer_u.compute_s_coarse is True


# ── S_coarse as an arbitration FACTOR (use_s_coarse) ─────────────────────

def test_use_s_coarse_off_leaves_score_unchanged():
    """use_s_coarse=False (default): the score is the plain rule, byte-identical
    even when s_coarse is being recorded (compute_s_coarse=True)."""
    q, t = _identical_clouds()
    pose = PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                          breakdown={"s_icp": 0.8, "R_coarse": np.eye(3),
                                     "t_coarse": np.array([10.0, 10.0, 10.0])})
    rec = ChampionScorer(compute_s_coarse=True).score(pose, q, t)
    assert rec.score == pytest.approx(0.8 * rec.breakdown["s_feat_1"])


def test_use_s_coarse_multiplies_score_by_clamped_s_coarse():
    q, t = _identical_clouds()
    # coarse pose aligned -> s_coarse ~1; refined aligned -> s_feat_1 ~1.
    pose = PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                          breakdown={"s_icp": 0.8, "R_coarse": np.eye(3),
                                     "t_coarse": np.zeros(3)})
    out = ChampionScorer(use_s_coarse=True).score(pose, q, t)
    sc = out.breakdown["s_coarse"]
    assert out.score == pytest.approx(0.8 * out.breakdown["s_feat_1"] * max(sc, 0.0))

    # a bad coarse pose (10 m off) -> s_coarse 0 -> score collapses to 0.
    bad = PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                         breakdown={"s_icp": 0.8, "R_coarse": np.eye(3),
                                    "t_coarse": np.array([10.0, 10.0, 10.0])})
    out_bad = ChampionScorer(use_s_coarse=True).score(bad, q, t)
    assert out_bad.breakdown["s_coarse"] == 0.0 and out_bad.score == 0.0


def test_use_s_coarse_requires_coarse_pose():
    q, t = _identical_clouds()
    pose = PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                          breakdown={"s_icp": 0.8})
    with pytest.raises(ValueError, match="coarse pose"):
        ChampionScorer(use_s_coarse=True).score(pose, q, t)
    # a HALF-stashed coarse pose (R_coarse but no t_coarse) is still a loud
    # ValueError, not a raw KeyError
    half = PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                          breakdown={"s_icp": 0.8, "R_coarse": np.eye(3)})
    with pytest.raises(ValueError, match="coarse pose"):
        ChampionScorer(use_s_coarse=True).score(half, q, t)


def test_use_s_coarse_matches_rule_replay_same_rule():
    """Fixture-level: ChampionScorer(use_s_coarse) final score EQUALS
    rule_replay's 's_icp*s_feat_1*metric_fit*s_coarse' evaluated on the SAME
    recorded term values — the arbitration term and the offline rule are one
    formula."""
    import importlib.util
    pd = pytest.importorskip("pandas")
    spec = importlib.util.spec_from_file_location(
        "rule_replay",
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "examples" / "rule_replay.py")
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)

    rng = np.random.default_rng(11)
    pts = rng.uniform(-0.05, 0.05, (60, 3))
    fq = rng.standard_normal((60, 8))
    ft = fq + 0.3 * rng.standard_normal((60, 8))     # so s_feat_1 in (0, 1)
    q = PointFeatures(pts=pts, feats=fq, meta={"feats_w1": fq})
    t = PointFeatures(pts=pts.copy(), feats=ft, meta={"feats_w1": ft})
    pose = PoseHypothesis(
        R=np.eye(3), t=np.zeros(3), score=0.0,
        breakdown={"s_icp": 0.77, "R_coarse": np.eye(3),
                   "t_coarse": np.array([0.004, 0.0, 0.0])})  # slight coarse offset
    out = ChampionScorer(size_aware=True, use_s_coarse=True).score(pose, q, t)

    df = pd.DataFrame([{k: out.breakdown[k] if k != "s_icp" else 0.77
                        for k in ("s_icp", "s_feat_1", "metric_fit", "s_coarse")}])
    terms = rr.parse_rule("s_icp*s_feat_1*metric_fit*s_coarse", df.columns)
    assert rr.rule_score(df, terms).iloc[0] == pytest.approx(out.score)


def test_icp_refiner_keep_coarse_stashes_pre_icp_pose():
    pytest.importorskip("open3d")
    from popoe.adapters import ICPRefiner
    from popoe.interfaces import ObjectModel, Scene
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.05, 0.05, (40, 3))
    q = PointFeatures(pts=pts, feats=rng.standard_normal((40, 8)))
    t = PointFeatures(pts=pts.copy(), feats=rng.standard_normal((40, 8)),
                      pts_dense=pts.copy())
    scene = Scene(rgb=np.zeros((2, 2, 3), np.uint8),
                  depth=np.zeros((2, 2), np.float32), K=np.eye(3))
    obj = ObjectModel(obj_id=1, mesh_path="x", diameter=0.1)
    Rc = np.eye(3); tc = np.array([0.01, 0.0, 0.0])
    pose = PoseHypothesis(R=Rc, t=tc, score=0.5, breakdown={"s_coarse": 0.5})

    off = ICPRefiner(tau_icp=0.03).refine(pose, scene, obj, q, t)
    assert "R_coarse" not in off.breakdown

    on = ICPRefiner(tau_icp=0.03, keep_coarse=True).refine(pose, scene, obj, q, t)
    assert np.array_equal(on.breakdown["R_coarse"], Rc)
    assert np.array_equal(on.breakdown["t_coarse"], tc)
