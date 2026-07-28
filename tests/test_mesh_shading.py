"""Which colour source a query mesh has, and what that does to the cache key.

The bug this guards: `_mesh_has_texture` asked only "is there a UV atlas", so a
mesh carrying `property uchar red/green/blue` — how BOP ships LM-O, TUD-L,
IC-BIN and HB — was rendered as flat beige and its DINOv2 query features were computed
on a colourless image.

The second half is the part that is easy to get wrong: fixing the renderer does
NOT change enc_cfg or the mesh bytes, so nothing in the old key moves and the
wrong features would be served from cache forever. The key must move for the
affected meshes and must NOT move for anything else.
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from popoe.cache import fingerprint  # noqa: E402
from popoe.freeze.feature_extractor import (  # noqa: E402
    SHADING_FACE_COLOR, SHADING_FLAT, SHADING_UV, SHADING_VERTEX_COLOR,
    mesh_shading_key_parts, resolve_mesh_shading,
)


def _box():
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


def test_plain_mesh_is_flat():
    assert resolve_mesh_shading(_box()) == SHADING_FLAT


def test_vertex_colours_are_detected():
    m = _box()
    m.visual.vertex_colors = np.tile(
        np.array([[200, 30, 40, 255]], dtype=np.uint8), (len(m.vertices), 1))
    assert m.visual.kind == "vertex"
    assert resolve_mesh_shading(m) == SHADING_VERTEX_COLOR


def test_face_colours_are_detected():
    m = _box()
    m.visual.face_colors = np.tile(
        np.array([[10, 220, 30, 255]], dtype=np.uint8), (len(m.faces), 1))
    assert m.visual.kind == "face"
    assert resolve_mesh_shading(m) == SHADING_FACE_COLOR


def test_uv_atlas_still_wins():
    """A textured mesh must keep taking the texture sampler, not the new
    vertex-colour branch — the two disagree about the v axis."""
    PIL = pytest.importorskip("PIL.Image")
    m = _box()
    m.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(m.vertices), 2), dtype=np.float64),
        image=PIL.new("RGB", (4, 4)))
    assert resolve_mesh_shading(m) == SHADING_UV


def test_uv_only_env_restores_the_old_behaviour(monkeypatch):
    """The A/B arm: one commit, two renderers. `uv-only` must make a
    vertex-coloured mesh look untextured again, exactly as before the fix."""
    m = _box()
    m.visual.vertex_colors = np.tile(
        np.array([[200, 30, 40, 255]], dtype=np.uint8), (len(m.vertices), 1))
    monkeypatch.setenv("POPOE_MESH_SHADING", "uv-only")
    assert resolve_mesh_shading(m) == SHADING_FLAT


# --- cache keys -----------------------------------------------------------

def _write(tmp_path, mesh, name):
    p = tmp_path / name
    mesh.export(p)
    return p


def test_key_parts_empty_for_unaffected_meshes(tmp_path):
    """A colourless mesh renders exactly as it did, so its key must not move —
    otherwise the fix silently discards every T-LESS/ITODD/YCB-V cache entry."""
    p = _write(tmp_path, _box(), "flat.ply")
    assert mesh_shading_key_parts(p) == ()


def test_key_parts_move_for_vertex_coloured_meshes(tmp_path):
    m = _box()
    m.visual.vertex_colors = np.tile(
        np.array([[200, 30, 40, 255]], dtype=np.uint8), (len(m.vertices), 1))
    p = _write(tmp_path, m, "vcol.ply")
    parts = mesh_shading_key_parts(p)
    assert parts == ("shading=vcolor-v1",)
    # ... and that must actually change the fingerprint the eval builds.
    cfg = {"grid": "32"}
    assert (fingerprint("query", cfg, "deadbeef", 1)
            != fingerprint("query", cfg, "deadbeef", 1, *parts))


def test_uv_only_reproduces_the_pre_fix_key(tmp_path, monkeypatch):
    """The old arm must HIT the old cache, or the A/B pays for two cold runs
    and stops being an A/B at one commit."""
    m = _box()
    m.visual.vertex_colors = np.tile(
        np.array([[200, 30, 40, 255]], dtype=np.uint8), (len(m.vertices), 1))
    p = _write(tmp_path, m, "vcol.ply")
    monkeypatch.setenv("POPOE_MESH_SHADING", "uv-only")
    assert mesh_shading_key_parts(p) == ()
