"""The paper-fidelity knobs: top-k correspondences and the env-driven
feature-side settings (FreeZeV2 Sec. IV-A).

What is guarded here and why:

  * corr_topk threading — a "faithful" run whose knob silently fails to reach
    the solver would carry k=10 in its provenance while matching at k=1, which
    is exactly the label-vs-measurement failure this project keeps re-learning.
  * the refusal on non-o3d solvers — a silent ignore is the same defect.
  * the correspondence path itself, on a synthetic instance where the true
    pose is known — not for accuracy (RANSAC on 60 points), but to prove the
    precomputed pairs reach Open3D in the right order: with [query, target]
    swapped, recovering the planted rotation from distinctive features is not
    possible, so a grossly wrong answer would expose the transposition.

The render-canvas / fill / min-views / canon-basis knobs live inside the GPU
extraction path and are exercised on the pod; their cache-key guard is the
conditional-add block in bop_eval (tested in test_bop_eval_cli, which asserts
default runs keep the historical key set).
"""

import numpy as np
import pytest

from popoe.freeze.recipes import _build_solver, stages_for_object
from popoe.interfaces import CanonFrame, PointFeatures
from popoe.solvers import Open3DFeatureRansacSolver


def test_corr_topk_reaches_the_o3d_solver():
    solver, _, _ = stages_for_object(0.1, corr_topk=10)
    assert solver.corr_topk == 10


def test_corr_topk_defaults_to_the_historical_path():
    solver, _, _ = stages_for_object(0.1)
    assert solver.corr_topk == 0


@pytest.mark.parametrize("name", ["gpu", "gpu-feat", "teaser"])
def test_corr_topk_refuses_non_o3d_solvers(name):
    with pytest.raises(ValueError, match="corr_topk"):
        _build_solver(name, tau=0.03, n_ransac=10, corr_topk=10)


def _synthetic_instance(n=60, seed=3):
    """A query cloud, its features, and the same cloud under a known pose.

    Features are the (noised) point coordinates themselves — distinctive
    enough that top-k feature matching recovers mostly-true correspondences,
    which is the property the solver contract needs."""
    rng = np.random.default_rng(seed)
    pts_q = rng.uniform(-0.05, 0.05, (n, 3))
    ang = 0.6
    R = np.array([[np.cos(ang), -np.sin(ang), 0],
                  [np.sin(ang), np.cos(ang), 0],
                  [0, 0, 1]])
    t = np.array([0.02, -0.01, 0.03])
    pts_t = pts_q @ R.T + t
    feats = np.hstack([pts_q, pts_q]).astype(np.float64)
    fq = feats + rng.normal(0, 1e-4, feats.shape)
    ft = feats + rng.normal(0, 1e-4, feats.shape)
    return pts_q, fq, pts_t, ft, R, t


def test_correspondence_path_recovers_a_planted_pose():
    pts_q, fq, pts_t, ft, R_true, t_true = _synthetic_instance()
    solver = Open3DFeatureRansacSolver(tau_inlier=0.005, max_iteration=2000,
                                       seed=11, corr_topk=3)
    hyps = solver.solve(PointFeatures(pts=pts_q, feats=fq),
                        PointFeatures(pts=pts_t, feats=ft),
                        CanonFrame(center=np.zeros(3), scale=1.0))
    assert hyps, "correspondence path returned no hypotheses"
    R, t = hyps[0].R, hyps[0].t
    # Orthonormality first — a transposed correspondence list would already
    # fail the angle bound, but a malformed estimate should say so plainly.
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    ang_err = np.degrees(np.arccos(np.clip((np.trace(R_true.T @ R) - 1) / 2,
                                           -1, 1)))
    assert ang_err < 5.0, f"rotation off by {ang_err:.1f} deg"
    assert np.linalg.norm(t - t_true) < 0.005


def test_probe_corr_stats_on_a_planted_pose():
    """Feature-quality ground truth on a synthetic instance.

    Query features = coordinates (perfectly discriminative), target = the same
    cloud under a known pose. Then every top-1 correspondence is true and all
    three rates must be 1.0. Scrambling the target features must send rate1
    towards chance — this pins that the function measures the FEATURES, not
    the geometry (a bug that compared clouds directly would still pass the
    first half)."""
    from examples.bop_eval import probe_corr_stats
    from popoe.interfaces import PointFeatures

    rng = np.random.default_rng(5)
    pts_q = rng.uniform(-0.05, 0.05, (300, 3))
    ang = 0.4
    R = np.array([[np.cos(ang), -np.sin(ang), 0],
                  [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    t = np.array([0.01, 0.02, -0.03])
    pts_t = pts_q @ R.T + t
    feats = np.hstack([pts_q, pts_q]).astype(np.float64)
    q = PointFeatures(pts=pts_q, feats=feats)
    tgt = PointFeatures(pts=pts_t, feats=feats + rng.normal(0, 1e-6, feats.shape))
    gts = [dict(R=R, t=t * 1000.0)]          # GT t in mm, as scene_gt ships it
    syms = [dict(R=np.eye(3), t=np.zeros(3))]

    r1, r10, reach, med1, tau_mm = probe_corr_stats(q, tgt, gts, syms, 0.10)
    assert r1 == 1.0 and reach == 1.0
    assert med1 < 1.0                        # sub-millimetre top-1 error
    assert tau_mm == 3.0                     # 3% of a 100mm diameter

    scrambled = PointFeatures(pts=pts_t, feats=rng.permutation(feats))
    r1s, _, _, _, _ = probe_corr_stats(q, scrambled, gts, syms, 0.10)
    assert r1s < 0.2, f"scrambled features still match: rate1={r1s}"


def test_probe_corr_requires_cache_and_explicit_backend(tmp_path):
    """The two probe guards are SystemExit, not silent fallbacks: a probe that
    quietly built features would measure a different cache than it claims."""
    import subprocess, sys, os
    env = dict(os.environ, PYTHONPATH="src")
    base = [sys.executable, "examples/bop_eval.py", "--bop", str(tmp_path),
            "--dataset", "lmo", "--detections", "x.json",
            "--out", str(tmp_path / "o.csv"), "--probe-corr",
            str(tmp_path / "p.csv")]
    r = subprocess.run(base + ["--render-backend", "nvdiffrast"],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "--cache" in (r.stderr + r.stdout)
    # --render-backend defaults to nvdiffrast, so the auto guard needs the
    # explicit value; a default-auto assumption here was the test's own bug.
    r = subprocess.run(base + ["--cache", str(tmp_path / "c"),
                               "--render-backend", "auto"],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "render-backend" in (r.stderr + r.stdout)


def test_n_restarts_reaches_the_solver_and_refuses_non_o3d():
    solver, _, _ = stages_for_object(0.1, n_restarts=5)
    assert solver.n_restarts == 5
    assert stages_for_object(0.1)[0].n_restarts == 1
    with pytest.raises(ValueError, match="n_restarts"):
        _build_solver("gpu", tau=0.03, n_ransac=10, n_restarts=5)
