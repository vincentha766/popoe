"""recipes.stages_for_object solver selection (o3d | gpu | gpu-feat | teaser).
Solver CONSTRUCTION is dep-light (torch/open3d/teaserpp import lazily inside
.solve), so this runs without them.
"""
import pytest

import numpy as np

import popoe
from popoe import Detection, ObjectModel, PointFeatures, PoseHypothesis, Scene
from popoe.freeze.recipes import make_correspondence_pipeline, stages_for_object
from popoe.solvers import GPURansacSolver, Open3DFeatureRansacSolver, TeaserSolver


def test_default_solver_is_open3d_unchanged():
    solver, _, _ = stages_for_object(0.1)
    assert isinstance(solver, Open3DFeatureRansacSolver)


def test_gpu_solver_geometric():
    solver, _, _ = stages_for_object(0.1, solver="gpu")
    assert isinstance(solver, GPURansacSolver) and solver.fitness == "geometric"


def test_gpu_feat_solver_feature():
    solver, _, _ = stages_for_object(0.1, solver="gpu-feat")
    assert isinstance(solver, GPURansacSolver) and solver.fitness == "feature"


def test_teaser_solver():
    solver, _, _ = stages_for_object(0.1, solver="teaser")
    assert isinstance(solver, TeaserSolver)


def test_open3d_seed_defaults_to_unseeded():
    """The evaluated mainline must keep Open3D's historical unseeded RNG:
    turning determinism on silently would shift every existing o3d number."""
    solver, _, _ = stages_for_object(0.1)
    assert solver.seed is None


def test_open3d_seed_is_settable():
    assert Open3DFeatureRansacSolver(seed=7).seed == 7


def test_seed_reaches_the_open3d_solver():
    """Before this, the seed knob existed but nothing in the evaluated path
    could set it — reachable only by constructing the solver by hand, so
    `bop_eval` runs could not be made reproducible at all."""
    solver, _, _ = stages_for_object(0.1, seed=7)
    assert solver.seed == 7


def test_seed_overrides_the_gpu_solvers_own_default():
    """gpu* are already deterministic (their default is 42); an explicit seed
    must win, and None must leave that default alone."""
    seeded, _, _ = stages_for_object(0.1, solver="gpu", seed=7)
    assert seeded.seed == 7
    default, _, _ = stages_for_object(0.1, solver="gpu")
    assert default.seed == 42


def test_teaser_ignores_the_seed():
    """TEASER++ has no RNG, so a seed is not applicable — it must not become a
    constructor error either."""
    solver, _, _ = stages_for_object(0.1, solver="teaser", seed=7)
    assert isinstance(solver, TeaserSolver)


def test_provenance_reports_the_effective_seed_per_solver():
    """Regression: a flat "UNSEEDED" for every seed=None run was a false claim
    in the provenance of a cited run — only o3d is genuinely unseeded by
    default. gpu* carry their own deterministic default; teaser has no RNG."""
    from popoe.freeze.recipes import solver_provenance

    assert "UNSEEDED" in solver_provenance("o3d", None)
    assert "seed=7 (seeded)" in solver_provenance("o3d", 7)
    # Not "deterministic": a seeded o3d run is reproducible to within the
    # measured 0.08 pt, not bit-for-bit (see solver_provenance). The word in
    # the provenance line is what a reader uses to decide whether a small
    # between-run difference can be read as signal.
    assert "deterministic" not in solver_provenance("o3d", 7)

    for gpu in ("gpu", "gpu-feat", "gpu-feat-dist"):
        line = solver_provenance(gpu, None)
        assert "UNSEEDED" not in line, line
        assert "seed=42 (seeded)" in line, line
        assert "seed=7 (seeded)" in solver_provenance(gpu, 7)

    for seed in (None, 7):
        line = solver_provenance("teaser", seed)
        assert "no RNG" in line and "UNSEEDED" not in line, line


def test_provenance_prints_corr_topk_so_o3d_variants_differ():
    """o3d and o3d + corr_topk=10 must not print the same provenance line.
    The line must carry the effective corr_topk for every o3d configuration."""
    from popoe.freeze.recipes import solver_provenance

    bare = solver_provenance("o3d", 42, corr_topk=0)
    topk = solver_provenance("o3d", 42, corr_topk=10)
    assert "corr_topk=0" in bare, bare
    assert "corr_topk=10" in topk, topk
    assert bare != topk
    # unseeded path too — configurations can still be unseeded
    assert "corr_topk=0" in solver_provenance("o3d", None, corr_topk=0)
    assert "corr_topk=10" in solver_provenance("o3d", None, corr_topk=10)


def test_provenance_prints_distance_check_for_gpu_solvers():
    """gpu-feat vs gpu-feat-dist must be distinguishable from the log line alone
    even if a future rename drops the suffix; the effective bit is what matters."""
    from popoe.freeze.recipes import solver_provenance

    for name in ("gpu", "gpu-feat"):
        line = solver_provenance(name, 42)
        assert "distance_check=0" in line, line
    dist = solver_provenance("gpu-feat-dist", 42)
    assert "distance_check=1" in dist, dist
    # teaser has neither knob
    assert "corr_topk" not in solver_provenance("teaser", 42)
    assert "distance_check" not in solver_provenance("teaser", 42)


def test_unknown_solver_raises():
    with pytest.raises(ValueError, match="solver must be"):
        stages_for_object(0.1, solver="bogus")


def test_solver_gets_object_scaled_tau():
    # tau = TAU_FRAC * extent; the gpu solver receives it as tau_inlier
    from popoe.freeze.recipes import TAU_FRAC
    solver, _, _ = stages_for_object(0.2, solver="gpu")
    assert solver.tau_inlier == pytest.approx(TAU_FRAC * 0.2)


def test_make_correspondence_pipeline_is_a_pose_method():
    class _Seg:
        def segment(self, scene, obj):
            return [Detection(np.ones((4, 4), bool), 0.9)]

    class _Q:
        def encode_query(self, obj):
            from popoe import CanonFrame
            return PointFeatures(np.zeros((6, 3), np.float32),
                                 np.ones((6, 4), np.float32),
                                 meta={"canon_frame": CanonFrame(3.0)})

    class _T:
        def encode_target(self, scene, det, obj, frame):
            return PointFeatures(np.zeros((6, 3), np.float32),
                                 np.ones((6, 4), np.float32))

    class _Solver:
        def solve(self, q, t, frame=None):
            return [PoseHypothesis(np.eye(3), np.zeros(3), 0.5)]

    pipe = make_correspondence_pipeline(_Seg(), _Q(), _T(), 0.1, topk=1)
    assert isinstance(pipe, popoe.PoseMethod)
    assert isinstance(pipe, popoe.Pipeline)
    assert isinstance(pipe.solver, Open3DFeatureRansacSolver)
    assert pipe.scorer is not None
    assert len(pipe.refiners) == 1

    class _Pass:
        def refine(self, pose, scene, obj, query, target):
            return pose

    class _Score:
        def score(self, pose, query, target):
            return pose

    pipe.solver = _Solver()
    pipe.refiners = [_Pass()]
    pipe.scorer = _Score()
    scene = Scene(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32),
                  np.eye(3))
    hyp = pipe.run(scene, ObjectModel(1, "a.ply", 0.1))
    assert hyp is not None
    assert np.allclose(hyp.R, np.eye(3))


def test_make_correspondence_pipeline_unwraps_rerank_chain():
    pipe = make_correspondence_pipeline(
        object(), object(), object(), 0.1, render_rerank=True)
    assert len(pipe.refiners) == 2


def test_make_correspondence_pipeline_matches_stages_for_object():
    """Eval and the factory share one wiring; fields must not drift."""
    kw = dict(size_aware=True, solver="gpu", seed=3, render_rerank=True,
              use_s_coarse=True, eq5_terms=True)
    solver, refiner, scorer = stages_for_object(0.15, **kw)
    pipe = make_correspondence_pipeline(
        object(), object(), object(), 0.15, **kw)
    assert type(pipe.solver) is type(solver)
    assert pipe.solver.seed == solver.seed
    assert pipe.solver.tau_inlier == solver.tau_inlier
    assert pipe.scorer.use_s_coarse is scorer.use_s_coarse
    assert pipe.scorer.size_aware is scorer.size_aware
    assert pipe.scorer.eq5_terms is scorer.eq5_terms
    assert len(pipe.refiners) == 2
    assert type(pipe.refiners[0]) is type(refiner.refiners[0])
