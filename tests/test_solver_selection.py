"""recipes.stages_for_object solver selection (o3d | gpu | gpu-feat | teaser).
Solver CONSTRUCTION is dep-light (torch/open3d/teaserpp import lazily inside
.solve), so this runs without them.
"""
import pytest

from popoe.freeze.recipes import stages_for_object
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


def test_unknown_solver_raises():
    with pytest.raises(ValueError, match="solver must be"):
        stages_for_object(0.1, solver="bogus")


def test_solver_gets_object_scaled_tau():
    # tau = TAU_FRAC * extent; the gpu solver receives it as tau_inlier
    from popoe.freeze.recipes import TAU_FRAC
    solver, _, _ = stages_for_object(0.2, solver="gpu")
    assert solver.tau_inlier == pytest.approx(TAU_FRAC * 0.2)
