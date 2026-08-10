"""POPOE_QUERY_SAMPLER (gedi arm D20).

The point of these is not that poisson sampling works — Open3D's does. It is
that the DEFAULT path is untouched and that the knob reaches the cache key.
A sampler swap that misses enc_cfg reuses the old cloud's features under the old
key and produces a run that looks entirely normal; there is no symptom to notice.
"""
import numpy as np
import pytest
import trimesh

from popoe.cache import conditional_enc_entries
from popoe.freeze.adapters import sample_query_surface, query_sampler_provenance


@pytest.fixture
def mesh_path(tmp_path):
    # mm-scale like a BOP model, and non-trivial enough for spacing to mean
    # something. icosphere is watertight, so both samplers are well-defined.
    m = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
    p = tmp_path / "obj_000001.ply"
    m.export(p)
    return str(p)


def test_default_is_bitwise_the_pre_d20_call(mesh_path, monkeypatch):
    """The knob must be invisible when unset: every published number, and every
    cached query feature on the GPU host, was produced by this exact call."""
    monkeypatch.delenv("POPOE_QUERY_SAMPLER", raising=False)
    got = sample_query_surface(mesh_path, 800, seed=7)
    mesh = trimesh.load(mesh_path, force="mesh")
    ref, _ = trimesh.sample.sample_surface_even(mesh, 800, seed=7)
    assert np.array_equal(got, np.asarray(ref))


def test_unset_and_explicit_even_agree(mesh_path, monkeypatch):
    monkeypatch.delenv("POPOE_QUERY_SAMPLER", raising=False)
    a = sample_query_surface(mesh_path, 500, seed=3)
    assert conditional_enc_entries().get("query_sampler") is None  # key unmoved
    monkeypatch.setenv("POPOE_QUERY_SAMPLER", "even")
    assert np.array_equal(a, sample_query_surface(mesh_path, 500, seed=3))


def test_poisson_enters_the_cache_key(monkeypatch):
    monkeypatch.setenv("POPOE_QUERY_SAMPLER", "poisson")
    assert conditional_enc_entries()["query_sampler"] == "poisson"


def test_poisson_is_reproducible_and_better_spaced(mesh_path, monkeypatch):
    """Open3D's sampler draws from a PROCESS-GLOBAL RNG — without the reseed in
    sample_query_surface, two calls differ and the arm is unreproducible."""
    from scipy.spatial import cKDTree

    monkeypatch.setenv("POPOE_QUERY_SAMPLER", "poisson")
    a = sample_query_surface(mesh_path, 500, seed=11)
    b = sample_query_surface(mesh_path, 500, seed=11)
    assert np.array_equal(a, b)

    monkeypatch.setenv("POPOE_QUERY_SAMPLER", "even")
    e = sample_query_surface(mesh_path, 500, seed=11)

    def min_nn(p):
        return cKDTree(p).query(p, k=2)[0][:, 1].min()

    # Gate G6 in gedi specs/D20_SAMPLER_ARM_SPEC.md. Measured 1.27-1.29x on six
    # YCB-V meshes at N=5000; 1.15 is the floor with margin.
    assert min_nn(a) / min_nn(e) >= 1.15


def test_typo_is_fatal_not_a_silent_default(mesh_path, monkeypatch):
    monkeypatch.setenv("POPOE_QUERY_SAMPLER", "poison")
    with pytest.raises(ValueError, match="even.*poisson"):
        sample_query_surface(mesh_path, 10, seed=1)


def test_provenance_names_the_effective_sampler(monkeypatch):
    monkeypatch.delenv("POPOE_QUERY_SAMPLER", raising=False)
    assert "query_sampler=even" in query_sampler_provenance(5000)
    monkeypatch.setenv("POPOE_QUERY_SAMPLER", "poisson")
    line = query_sampler_provenance(5000)
    assert "query_sampler=poisson" in line and "n_points=5000" in line
