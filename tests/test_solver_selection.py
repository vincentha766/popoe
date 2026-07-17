"""recipes.stages_for_object solver selection (o3d | gpu | gpu-feat). Solver
CONSTRUCTION is dep-light (torch/open3d import lazily inside .solve), so this
runs without them.
"""
import pytest

from popoe.recipes import stages_for_object
from popoe.solvers import GPURansacSolver, Open3DFeatureRansacSolver


def test_default_solver_is_open3d_unchanged():
    solver, _, _ = stages_for_object(0.1)
    assert isinstance(solver, Open3DFeatureRansacSolver)


def test_gpu_solver_geometric():
    solver, _, _ = stages_for_object(0.1, solver="gpu")
    assert isinstance(solver, GPURansacSolver) and solver.fitness == "geometric"


def test_gpu_feat_solver_feature():
    solver, _, _ = stages_for_object(0.1, solver="gpu-feat")
    assert isinstance(solver, GPURansacSolver) and solver.fitness == "feature"


def test_unknown_solver_raises():
    with pytest.raises(ValueError, match="solver must be"):
        stages_for_object(0.1, solver="bogus")


def test_solver_gets_object_scaled_tau():
    # tau = TAU_FRAC * extent; the gpu solver receives it as tau_inlier
    from popoe.recipes import TAU_FRAC
    solver, _, _ = stages_for_object(0.2, solver="gpu")
    assert solver.tau_inlier == pytest.approx(TAU_FRAC * 0.2)
