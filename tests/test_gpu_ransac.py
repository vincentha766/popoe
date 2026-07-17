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


def test_feature_fitness_recovers_known_pose():
    q, t, R, tt = _matched_cloud(n=150, seed=12)
    h = GPURansacSolver(tau_inlier=0.003, iters=10000, device="cpu", seed=0,
                        fitness="feature").solve(q, t, _frame())
    assert h, "feature fitness should recover"
    ang, terr = _err(R, tt, h[0].R, h[0].t)
    assert ang < 2.0 and terr < 3e-3


def _unit(rng, n, d):
    v = rng.standard_normal((n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _adversarial_fixture(seed=0):
    """MANY true correspondences at MODERATE cosine (support pose A) + a FEW
    high-cosine decoys (support a different pose B). Under Eq.5 with a FIXED |P_T|
    denominator, pose A wins on inlier QUANTITY x quality; under the buggy
    normalize-by-inlier-count (mean cosine), the few near-1.0 decoys make pose B
    win. Returns clouds, the correspondence pool, and both poses."""
    rng = np.random.default_rng(seed)
    D, n_true, n_dec = 128, 90, 10
    RA, tA = _rigid(seed + 1)                         # true pose (many support)
    RB, tB = _rigid(seed + 2)                         # decoy pose (few support)

    q_true = rng.uniform(-0.06, 0.06, (n_true, 3))
    t_true = q_true @ RA.T + tA
    q_dec = rng.uniform(-0.06, 0.06, (n_dec, 3))
    t_dec = q_dec @ RB.T + tB

    # true feats: cosine ~0.4 per matched pair, ~0 across pairs (random basis).
    u = _unit(rng, n_true, D)
    nz = _unit(rng, n_true, D)
    ft_true = 0.4 * u + np.sqrt(1 - 0.16) * nz
    ft_true /= np.linalg.norm(ft_true, axis=1, keepdims=True)
    # decoy feats: query == target (cosine 1.0), distinct from the true basis.
    w = _unit(rng, n_dec, D)

    pts_q = np.vstack([q_true, q_dec])
    pts_t = np.vstack([t_true, t_dec])
    fq = np.vstack([u, w])
    ftt = np.vstack([ft_true, w.copy()])
    return pts_q, pts_t, fq, ftt, (RA, tA), (RB, tB)


def _pool_scores(pts_q, pts_t, fq, ft, R, t, thr, k=10):
    """Replicate the solver's Eq.5 fixed-|P_T| score AND the buggy mean-cosine
    score for one pose, over the top-k correspondence pool."""
    fqn = fq / np.linalg.norm(fq, axis=1, keepdims=True)
    ftn = ft / np.linalg.norm(ft, axis=1, keepdims=True)
    sim = ftn @ fqn.T
    topi = np.argsort(-sim, axis=1)[:, :k]
    c_t = np.repeat(np.arange(len(pts_t)), k)
    c_q = topi.reshape(-1)
    c_sim = sim[c_t, c_q]
    moved = pts_q[c_q] @ R.T + t
    d = np.linalg.norm(moved - pts_t[c_t], axis=1)
    inl = d < thr
    n_in = int(inl.sum())
    fixed = float((c_sim * inl).sum() / len(pts_t))       # Eq.5 (correct)
    mean_cos = float((c_sim * inl).sum() / max(n_in, 1))  # the -31pt bug
    return fixed, mean_cos, n_in


def test_feature_fitness_fixed_denominator_not_hijacked():
    """The ch3 lesson: clean synthetic features don't surface Eq.5 bugs — an
    ADVERSARIAL similarity structure does. The solver (fixed |P_T|) must recover
    the MANY-true-correspondence pose, not the few-high-cosine-decoy pose."""
    pts_q, pts_t, fq, ft, (RA, tA), (RB, tB) = _adversarial_fixture(seed=3)
    q = PointFeatures(pts=pts_q, feats=fq, meta={"feats_w1": fq})
    t = PointFeatures(pts=pts_t, feats=ft, meta={"feats_w1": ft})
    thr = 0.004

    # The fixture is genuinely a trap: mean-cosine ranks the DECOY pose ABOVE the
    # true pose, while the fixed-|P_T| Eq.5 score ranks the true pose above.
    fA, mA, nA = _pool_scores(pts_q, pts_t, fq, ft, RA, tA, thr)
    fB, mB, nB = _pool_scores(pts_q, pts_t, fq, ft, RB, tB, thr)
    assert nB >= 6 and nA > nB                    # decoys pass min_inliers; A has more
    assert mB > mA                                # mean-cosine WOULD be hijacked
    assert fA > fB                                # fixed |P_T| is not

    # the solver, using the fixed denominator, recovers pose A (not B).
    h = GPURansacSolver(tau_inlier=thr, iters=20000, device="cpu", seed=0,
                        fitness="feature", min_inliers=6).solve(q, t, _frame())
    assert h, "should recover the true pose"
    angA, _ = _err(RA, tA, h[0].R, h[0].t)
    angB, _ = _err(RB, tB, h[0].R, h[0].t)
    assert angA < 3.0 and angB > 20.0            # locked onto A, far from B

    # Pin the IMPLEMENTED denominator (recovering A alone wouldn't — geometric
    # count also picks A): the reported gpu_score must equal the fixed-|P_T|
    # score at the recovered pose, and NOT the mean-cosine (n_in) score.
    fixed_h, mean_h, n_h = _pool_scores(pts_q, pts_t, fq, ft, h[0].R, h[0].t, thr)
    assert h[0].breakdown["gpu_score"] == pytest.approx(fixed_h, rel=1e-2)
    assert abs(h[0].breakdown["gpu_score"] - mean_h) > 0.02 * mean_h  # != mean-cosine
    # and NOT the geometric value (n_in / N_t) either
    assert abs(h[0].breakdown["gpu_score"] - n_h / len(pts_t)) > 0.05


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
