"""TeaserSolver — TEASER++ certifiable registration as a PoseSolver.

Mirrors test_gpu_ransac's fixture conventions: recovers a known pose from
synthetic distinctive-feature correspondences, returns the shared hypothesis
shape, reads w=1 features, and survives an outlier ratio that stresses RANSAC.

Needs teaserpp_python (source build — skipped when absent). No GPU.
"""
import numpy as np
import pytest

pytest.importorskip("teaserpp_python")

from popoe.interfaces import CanonFrame, PointFeatures
from popoe.solvers.teaser import TeaserSolver


def _frame():
    return CanonFrame(center=np.zeros(3), scale=1.0)


def _rigid(seed=0):
    """A random-ish rigid transform (R query->target, t)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q, rng.uniform(-0.1, 0.1, 3)


def _matched_cloud(n=120, seed=1, noise=0.0):
    """Query points + their images under a known pose, with DISTINCTIVE per-point
    features (so top-1 NN gives the identity correspondence)."""
    rng = np.random.default_rng(seed)
    pts_q = rng.uniform(-0.06, 0.06, (n, 3)).astype(np.float64)
    R, t = _rigid(seed + 5)
    pts_t = pts_q @ R.T + t
    if noise:
        pts_t = pts_t + rng.normal(0, noise, pts_t.shape)
    feats = rng.standard_normal((n, 16)).astype(np.float64)  # unique per point
    q = PointFeatures(pts=pts_q, feats=feats, meta={"feats_w1": feats})
    t_pf = PointFeatures(pts=pts_t, feats=feats.copy(), meta={"feats_w1": feats.copy()})
    return q, t_pf, R, t


def _err(Ra, ta, Rb, tb):
    ang = np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))
    return ang, float(np.linalg.norm(ta - tb))


def test_recovers_known_pose():
    q, t, R, tt = _matched_cloud(n=150, seed=2)
    hyps = TeaserSolver(tau_inlier=0.003).solve(q, t, _frame())
    assert len(hyps) == 1
    ang, terr = _err(R, tt, hyps[0].R, hyps[0].t)
    assert ang < 1.0 and terr < 2e-3            # clean data: near-exact

def test_returns_shared_hypothesis_shape():
    q, t, *_ = _matched_cloud(seed=3)
    h = TeaserSolver().solve(q, t, _frame())[0]
    assert h.R.shape == (3, 3) and h.t.shape == (3,)
    # same breakdown key the other solvers / downstream stages rely on
    assert "s_coarse" in h.breakdown and h.score == h.breakdown["s_coarse"]
    assert h.breakdown["n_corr"] >= 3
    assert h.breakdown["n_clique"] >= 3
    # rotation is orthonormal with det +1
    assert np.allclose(h.R @ h.R.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(h.R) == pytest.approx(1.0, abs=1e-6)


def test_uses_w1_features_not_dot_feats():
    """The solver must correspond/score in the w=1 space (meta['feats_w1']). Here
    .feats are GARBAGE (uncorrelated across query/target) while feats_w1 holds the
    real matches — recovery proves feats_w1 is used, not .feats."""
    q, t, R, tt = _matched_cloud(n=150, seed=8)
    rng = np.random.default_rng(99)
    q = PointFeatures(pts=q.pts, feats=rng.standard_normal(q.feats.shape),
                      meta={"feats_w1": q.meta["feats_w1"]})
    t = PointFeatures(pts=t.pts, feats=rng.standard_normal(t.feats.shape),
                      meta={"feats_w1": t.meta["feats_w1"]})
    h = TeaserSolver(tau_inlier=0.003).solve(q, t, _frame())
    assert h, "should recover using feats_w1"
    ang, terr = _err(R, tt, h[0].R, h[0].t)
    assert ang < 1.0 and terr < 2e-3


def test_deterministic():
    """TEASER++ has no RNG — two runs must agree exactly."""
    q, t, *_ = _matched_cloud(seed=4)
    a = TeaserSolver().solve(q, t, _frame())[0]
    b = TeaserSolver().solve(q, t, _frame())[0]
    assert np.array_equal(a.R, b.R) and np.array_equal(a.t, b.t)


def test_robust_to_majority_outlier_correspondences():
    """TEASER++'s selling point: recovery when MOST feature matches are wrong.
    2/3 of the target points get features that match the WRONG query point
    (shuffled), so the top-1 correspondence set is ~67% outliers."""
    rng = np.random.default_rng(21)
    n = 150
    q, t, R, tt = _matched_cloud(n=n, seed=21)
    ft = t.meta["feats_w1"].copy()
    bad = rng.choice(n, size=100, replace=False)
    ft[bad] = ft[rng.permutation(bad)]           # wrong-but-plausible matches
    t = PointFeatures(pts=t.pts, feats=ft, meta={"feats_w1": ft})
    h = TeaserSolver(tau_inlier=0.003).solve(q, t, _frame())
    assert h, "should survive 67% outlier correspondences"
    ang, terr = _err(R, tt, h[0].R, h[0].t)
    assert ang < 2.0 and terr < 3e-3


def test_noise_tolerance():
    q, t, R, tt = _matched_cloud(n=150, seed=6, noise=0.002)
    h = TeaserSolver(tau_inlier=0.01).solve(q, t, _frame())
    assert h
    ang, terr = _err(R, tt, h[0].R, h[0].t)
    assert ang < 5.0 and terr < 1e-2


def test_aborted_solve_returns_empty_not_garbage():
    """codex-review finding: the 1.1.0 binding's solution has no `.valid`, and an
    internally-aborted TEASER solve ("Max clique lower bound equals to zero")
    still hands back an uninitialized R/t — which is shape-correct and would ride
    through ICP/scoring as a real pose. Pairwise-INCONSISTENT correspondences
    (query spread ~1 m, target spread ~1 mm, noise bound 3 mm: no two TIMs can
    agree) force a max clique < 3, which must yield NO hypothesis."""
    rng = np.random.default_rng(5)
    n = 40
    pts_q = rng.uniform(-0.5, 0.5, (n, 3))
    pts_t = rng.uniform(-0.0005, 0.0005, (n, 3))
    feats = rng.standard_normal((n, 16))
    q = PointFeatures(pts=pts_q, feats=feats, meta={"feats_w1": feats})
    t = PointFeatures(pts=pts_t, feats=feats.copy(), meta={"feats_w1": feats.copy()})
    assert TeaserSolver(tau_inlier=0.003).solve(q, t, _frame()) == []


def test_degenerate_returns_empty():
    q = PointFeatures(pts=np.zeros((2, 3)), feats=np.zeros((2, 16)),
                      meta={"feats_w1": np.zeros((2, 16))})
    assert TeaserSolver().solve(q, q, _frame()) == []


def test_max_corr_caps_pool():
    q, t, R, tt = _matched_cloud(n=150, seed=9)
    h = TeaserSolver(tau_inlier=0.003, max_corr=80).solve(q, t, _frame())
    assert h and h[0].breakdown["n_corr"] <= 80
    ang, terr = _err(R, tt, h[0].R, h[0].t)  # still recovers from the capped pool
    assert ang < 1.0 and terr < 2e-3


def test_agrees_with_open3d_solver_within_tolerance():
    pytest.importorskip("open3d")
    from popoe.solvers.open3d_ransac import Open3DFeatureRansacSolver
    q, t, R, tt = _matched_cloud(n=140, seed=6)
    teaser = TeaserSolver(tau_inlier=0.01).solve(q, t, _frame())[0]
    o3d = Open3DFeatureRansacSolver(tau_inlier=0.01, max_iteration=5000).solve(q, t, _frame())
    assert o3d, "open3d solver should return a hypothesis"
    # both recover the same known pose -> agree with each other
    ang, terr = _err(teaser.R, teaser.t, o3d[0].R, o3d[0].t)
    assert ang < 2.0 and terr < 3e-3
