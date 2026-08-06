"""Unit tests for render_rerank pure helpers (no GPU)."""
import numpy as np
import pytest

from popoe.render_rerank import (
    bbox_from_mask,
    pca_flip_variants,
    pick_by_scores,
    rot_about,
    RenderAppearanceReranker,
)
from popoe.interfaces import PoseHypothesis, PointFeatures, Scene, ObjectModel


def test_rot_about_180_is_involutory():
    R = rot_about([0, 0, 1], 180)
    assert np.allclose(R @ R, np.eye(3), atol=1e-9)


def test_pca_flip_variants_set():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 3))
    pts[:, 2] *= 3  # elongated
    R0 = np.eye(3)
    vs = pca_flip_variants(R0, pts, include_azimuth=True)
    names = [n for n, _ in vs]
    assert names[0] == "champion"
    assert "flip0" in names and "flip1" in names and "flip2" in names
    assert "az90" in names and "az270" in names
    assert len(vs) == 6  # champion + 3 flips + 2 az


def test_pca_flip_variants_no_azimuth():
    pts = np.eye(3)
    vs = pca_flip_variants(np.eye(3), pts, include_azimuth=False)
    assert len(vs) == 4  # champion + 3 flips


def test_pick_by_scores_argmax():
    R0, R1 = np.eye(3), rot_about([0, 1, 0], 180)
    vs = [("champion", R0), ("flip0", R1)]
    name, R, s = pick_by_scores(vs, {"champion": 0.1, "flip0": 0.9})
    assert name == "flip0"
    assert s == 0.9
    assert np.allclose(R, R1)


def test_bbox_from_mask():
    m = np.zeros((10, 12), dtype=bool)
    m[2:5, 3:7] = True
    assert bbox_from_mask(m) == (3, 2, 7, 5)
    assert bbox_from_mask(np.zeros((4, 4), dtype=bool)) is None


def test_reranker_disabled_is_identity():
    rr = RenderAppearanceReranker(enabled=False)
    pose = PoseHypothesis(np.eye(3), np.zeros(3), 0.5, {"k": 1})
    scene = Scene(rgb=np.zeros((8, 8, 3), np.uint8),
                  depth=np.zeros((8, 8), np.float32),
                  K=np.eye(3))
    obj = ObjectModel(obj_id=1, mesh_path="/tmp/x.ply", diameter=0.1)
    q = PointFeatures(pts=np.eye(3), feats=np.eye(3))
    t = PointFeatures(pts=np.eye(3), feats=np.eye(3))
    out = rr.refine(pose, scene, obj, q, t)
    assert out.score == 0.5
    assert np.allclose(out.R, pose.R)


def _icp_scene(n=400, seed=0):
    """A query cloud, an identical dense target, and the surrounding stubs.

    Clouds are metres with ~0.1 m extent — the real BOP scale, which is the
    whole point: 3% of it is ~3 mm, and the regression being guarded is a stage
    that used 0.03 *metres* instead.
    """
    rng = np.random.default_rng(seed)
    pts = rng.normal(scale=0.03, size=(n, 3))
    q = PointFeatures(pts=pts, feats=rng.normal(size=(n, 4)))
    t = PointFeatures(pts=pts.copy(), feats=rng.normal(size=(n, 4)),
                      pts_dense=pts.copy(), meta={"bbox": (0, 0, 8, 8)})
    scene = Scene(rgb=np.zeros((8, 8, 3), np.uint8),
                  depth=np.zeros((8, 8), np.float32), K=np.eye(3))
    obj = ObjectModel(obj_id=1, mesh_path="/tmp/x.ply", diameter=0.1)
    return q, t, scene, obj


def test_icp_refiner_records_its_tau():
    """Whoever re-runs ICP downstream needs the tau, and must not guess it."""
    from popoe.adapters import ICPRefiner
    q, t, scene, obj = _icp_scene()
    ref = ICPRefiner(tau_icp=0.0031)
    out = ref.refine(PoseHypothesis(np.eye(3), np.zeros(3), 1.0, {}),
                     scene, obj, q, t)
    assert out.breakdown["tau_icp"] == pytest.approx(0.0031)


def test_tau_icp_reuses_recorded_value_not_a_default():
    q, _, _, _ = _icp_scene()
    pose = PoseHypothesis(np.eye(3), np.zeros(3), 1.0, {"tau_icp": 0.0031})
    assert RenderAppearanceReranker._tau_icp(pose, q) == pytest.approx(0.0031)


def test_tau_icp_fallback_is_metres_scaled_not_0_03():
    """No recorded tau → 3% of extent. The old code's 0.03 default was 30 mm,
    ~10x too loose on a 0.1 m object, which inflated fitness ~4x."""
    q, _, _, _ = _icp_scene()
    pose = PoseHypothesis(np.eye(3), np.zeros(3), 1.0, {})
    tau = RenderAppearanceReranker._tau_icp(pose, q)
    extent = float(np.ptp(q.pts, axis=0).max())
    assert tau == pytest.approx(0.03 * extent)
    assert tau < 0.03 / 2      # i.e. nowhere near the old 0.03 m default


def test_champion_is_re_icped_too(monkeypatch):
    """Every variant must be measured the same way.

    Re-ICP'ing only the flipped variants left them with a fitness from a second
    ICP pass while the champion kept the first one — and s_icp reaches
    ChampionScorer as a multiplicative factor, so flips won on measurement
    asymmetry alone (LM-O AR(2/3) 0.77 -> 0.25 on the 2026-07-30 two-line runs).
    Here the champion WINS the appearance vote, and re-ICP must still have run.
    """
    q, t, scene, obj = _icp_scene()
    rr = RenderAppearanceReranker(re_icp=True)
    monkeypatch.setattr(
        rr, "_sar_ti",
        lambda scene, obj, R, t_m, bbox: 1.0 if np.allclose(R, np.eye(3)) else 0.0)
    pose = PoseHypothesis(np.eye(3), np.zeros(3), 1.0,
                          {"s_icp": 0.11, "tau_icp": 0.03 * 0.1})
    out = rr.refine(pose, scene, obj, q, t)
    assert out.breakdown["render_rerank"] == "champion"
    assert out.breakdown["render_rerank_re_icp"] is True
    # And the re-measured fitness replaced the stale one it was handed.
    assert out.breakdown["s_icp"] != pytest.approx(0.11)


def test_re_icp_uses_recorded_tau(monkeypatch):
    """A 10x-too-loose tau shows up directly as an inflated inlier ratio."""
    from popoe.registration import icp_refinement
    q, t, scene, obj = _icp_scene()
    seen = []
    monkeypatch.setattr("popoe.registration.icp_refinement",
                        lambda *a, **kw: (seen.append(kw["tau_icp"]),
                                          icp_refinement(*a, **kw))[1])
    rr = RenderAppearanceReranker(re_icp=True)
    monkeypatch.setattr(rr, "_sar_ti",
                        lambda scene, obj, R, t_m, bbox: 1.0)
    rr.refine(PoseHypothesis(np.eye(3), np.zeros(3), 1.0, {"tau_icp": 0.0031}),
              scene, obj, q, t)
    assert seen == [pytest.approx(0.0031)]


def test_reranker_skips_without_bbox():
    rr = RenderAppearanceReranker(enabled=True)
    # Force enabled path but no bbox → skip without loading GPU backends.
    pose = PoseHypothesis(np.eye(3), np.zeros(3), 0.42, {})
    scene = Scene(rgb=np.zeros((8, 8, 3), np.uint8),
                  depth=np.zeros((8, 8), np.float32),
                  K=np.eye(3))
    obj = ObjectModel(obj_id=1, mesh_path="/tmp/x.ply", diameter=0.1)
    q = PointFeatures(pts=np.random.randn(20, 3), feats=np.random.randn(20, 4))
    t = PointFeatures(pts=np.random.randn(20, 3), feats=np.random.randn(20, 4),
                      meta={})  # no detection / bbox
    out = rr.refine(pose, scene, obj, q, t)
    assert out.breakdown.get("render_rerank") == "skipped_no_bbox"
    assert out.score == 0.42
    assert np.allclose(out.breakdown["R_prererank"], pose.R)
    assert np.allclose(out.breakdown["t_prererank"], pose.t)


def test_flip_winner_corrects_coarse_pose(monkeypatch):
    """B1-F2: when a flip wins, the SAME model-frame delta must reach the
    coarse pose in the breakdown — otherwise ChampionScorer's S_coarse
    measures the pre-flip orientation and penalises the corrected candidate."""
    from popoe.render_rerank import pca_flip_variants
    q, t, scene, obj = _icp_scene()
    rr = RenderAppearanceReranker(re_icp=True)
    monkeypatch.setattr(
        rr, "_sar_ti",
        lambda scene, obj, R, t_m, bbox: 0.0 if np.allclose(R, np.eye(3)) else 1.0)
    R_coarse0 = rot_about(np.array([0.0, 0.0, 1.0]), 30.0)
    pose = PoseHypothesis(np.eye(3), np.zeros(3), 1.0,
                          {"s_icp": 0.11, "tau_icp": 0.03 * 0.1,
                           "R_coarse": R_coarse0.copy(),
                           "t_coarse": np.zeros(3)})
    out = rr.refine(pose, scene, obj, q, t)
    assert out.breakdown["render_rerank"] == "flip0"
    # pose.R = I, so the model-frame delta IS the flip0 variant rotation.
    flip0 = dict(pca_flip_variants(np.eye(3), q.pts))["flip0"]
    assert np.allclose(out.breakdown["R_coarse"], R_coarse0 @ flip0)
    assert np.allclose(out.breakdown["t_coarse"], 0.0)   # translation untouched


def test_champion_winner_leaves_coarse_untouched(monkeypatch):
    q, t, scene, obj = _icp_scene()
    rr = RenderAppearanceReranker(re_icp=True)
    monkeypatch.setattr(
        rr, "_sar_ti",
        lambda scene, obj, R, t_m, bbox: 1.0 if np.allclose(R, np.eye(3)) else 0.0)
    R_coarse0 = rot_about(np.array([0.0, 0.0, 1.0]), 30.0)
    pose = PoseHypothesis(np.eye(3), np.zeros(3), 1.0,
                          {"s_icp": 0.11, "tau_icp": 0.03 * 0.1,
                           "R_coarse": R_coarse0.copy()})
    out = rr.refine(pose, scene, obj, q, t)
    assert out.breakdown["render_rerank"] == "champion"
    assert np.allclose(out.breakdown["R_coarse"], R_coarse0)


def test_re_icp_failure_reverts_the_flip(monkeypatch):
    """A flipped R carrying the champion's pre-flip s_icp is not a scoreable
    candidate; without a consistent re-measurement the champion pose stays."""
    q, t, scene, obj = _icp_scene()
    rr = RenderAppearanceReranker(re_icp=True)
    monkeypatch.setattr(
        rr, "_sar_ti",
        lambda scene, obj, R, t_m, bbox: 0.0 if np.allclose(R, np.eye(3)) else 1.0)

    def boom(*a, **kw):
        raise RuntimeError("icp exploded")
    monkeypatch.setattr("popoe.registration.icp_refinement", boom)
    R_coarse0 = rot_about(np.array([0.0, 0.0, 1.0]), 30.0)
    pose = PoseHypothesis(np.eye(3), np.zeros(3), 1.0,
                          {"s_icp": 0.11, "tau_icp": 0.03 * 0.1,
                           "R_coarse": R_coarse0.copy()})
    out = rr.refine(pose, scene, obj, q, t)
    assert out.breakdown["render_rerank"] == "reverted:flip0"
    assert out.breakdown["render_rerank_re_icp"].startswith("failed:")
    assert np.allclose(out.R, np.eye(3))                   # champion pose kept
    assert out.breakdown["s_icp"] == pytest.approx(0.11)   # consistent pair
    assert np.allclose(out.breakdown["R_coarse"], R_coarse0)
