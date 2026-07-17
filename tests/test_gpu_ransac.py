"""GPURansacSolver (B layer) — the ported batched RANSAC, on CPU.

Block 1 covers the pure GEOMETRIC port: it recovers a known pose from synthetic
correspondences, returns the Open3D-solver-shaped hypothesis, and agrees with
Open3DFeatureRansacSolver within tolerance on a shared fixture.

Needs torch (CPU is fine). No GPU.
"""
import numpy as np
import pytest

pytest.importorskip("torch")

from popoe.interfaces import CanonFrame, PointFeatures
from popoe.solvers.gpu_ransac import GPURansacSolver


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


def test_geometric_recovers_known_pose():
    q, t, R, tt = _matched_cloud(n=150, seed=2)
    # tight tau: only near-exact poses score full inliers on clean data.
    solver = GPURansacSolver(tau_inlier=0.003, iters=10000, device="cpu", seed=0)
    hyps = solver.solve(q, t, _frame())
    assert len(hyps) == 1
    ang, terr = _err(R, tt, hyps[0].R, hyps[0].t)
    assert ang < 2.0 and terr < 3e-3            # recovers the known pose


def test_returns_open3d_shaped_hypothesis():
    q, t, *_ = _matched_cloud(seed=3)
    h = GPURansacSolver(iters=1000, device="cpu").solve(q, t, _frame())[0]
    assert h.R.shape == (3, 3) and h.t.shape == (3,)
    # same breakdown key the Open3D solver / downstream stages rely on
    assert "s_coarse" in h.breakdown and h.score == h.breakdown["s_coarse"]
    assert h.breakdown["fitness_mode"] == "geometric"
    assert h.breakdown["n_inliers"] >= 6


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
    h = GPURansacSolver(tau_inlier=0.003, iters=10000, device="cpu", seed=0).solve(q, t, _frame())
    assert h, "should recover using feats_w1"
    ang, terr = _err(R, tt, h[0].R, h[0].t)
    assert ang < 2.0 and terr < 3e-3


def test_iters_zero_returns_empty():
    q, t, *_ = _matched_cloud(seed=10)
    assert GPURansacSolver(iters=0, device="cpu").solve(q, t, _frame()) == []


def test_deterministic_given_seed():
    q, t, *_ = _matched_cloud(seed=4)
    a = GPURansacSolver(iters=1500, device="cpu", seed=7).solve(q, t, _frame())[0]
    b = GPURansacSolver(iters=1500, device="cpu", seed=7).solve(q, t, _frame())[0]
    assert np.array_equal(a.R, b.R) and np.array_equal(a.t, b.t)


def test_feature_fitness_not_available_in_this_step():
    with pytest.raises(ValueError, match="not available yet"):
        GPURansacSolver(fitness="feature")


def test_degenerate_returns_empty():
    q = PointFeatures(pts=np.zeros((2, 3)), feats=np.zeros((2, 16)),
                      meta={"feats_w1": np.zeros((2, 16))})
    assert GPURansacSolver(device="cpu").solve(q, q, _frame()) == []


def test_agrees_with_open3d_solver_within_tolerance():
    pytest.importorskip("open3d")
    from popoe.solvers.open3d_ransac import Open3DFeatureRansacSolver
    q, t, R, tt = _matched_cloud(n=140, seed=6)
    gpu = GPURansacSolver(tau_inlier=0.01, iters=10000, device="cpu", seed=1).solve(q, t, _frame())[0]
    o3d = Open3DFeatureRansacSolver(tau_inlier=0.01, max_iteration=5000).solve(q, t, _frame())
    assert o3d, "open3d solver should return a hypothesis"
    # both recover the same known pose -> agree with each other
    ang, terr = _err(gpu.R, gpu.t, o3d[0].R, o3d[0].t)
    assert ang < 2.0 and terr < 3e-3
