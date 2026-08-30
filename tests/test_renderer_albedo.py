"""Renders must carry the mesh's own colour.

Templates are matched against real RGB crops by DINOv2, so a silently
colourless render costs most of the similarity signal. These pin the contract
that used to be broken: colour reaches the image when the mesh has it, and the
beige fallback applies only when it genuinely has none.
"""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from popoe.renderer import (TrimeshRenderer, load_mesh_albedo,
                            load_mesh_for_rendering)


def _box(tmp_path, name="m.ply", color=None):
    m = trimesh.creation.box(extents=(60, 40, 25))
    if color is not None:
        m.visual.vertex_colors = np.tile(
            np.array(list(color) + [255], np.uint8), (len(m.vertices), 1))
    p = tmp_path / name
    m.export(p)
    return str(p)


def _render(mesh_path, albedo):
    V, F, N, _, _ = load_mesh_for_rendering(mesh_path)
    radius = float(np.linalg.norm(np.ptp(V, axis=0))) * 2.0
    cam = np.array([radius * 0.6, radius * 0.5, radius * 0.6], np.float32)
    rgb, depth = TrimeshRenderer(96, 96).render(
        V, F, cam, fov_deg=60.0, normals=N, albedo=albedo)
    return rgb, depth > 0


def test_vertex_colour_mesh_resolves_an_albedo(tmp_path):
    path = _box(tmp_path, color=(200, 30, 30))
    V, F, _, _, _ = load_mesh_for_rendering(path)
    alb = load_mesh_albedo(path, V, F)
    assert alb is not None and alb.shape == (len(V), 3)
    assert alb.max() <= 1.0 and alb.min() >= 0.0
    # red channel dominates, and it is not the beige fallback
    assert alb[:, 0].mean() > alb[:, 1].mean() + 0.3


def test_colourless_mesh_has_no_albedo(tmp_path):
    path = _box(tmp_path)
    V, F, _, _, _ = load_mesh_for_rendering(path)
    assert load_mesh_albedo(path, V, F) is None


def test_render_shows_the_mesh_colour(tmp_path):
    """The regression that motivated this: a red box rendering beige."""
    path = _box(tmp_path, color=(200, 30, 30))
    V, F, _, _, _ = load_mesh_for_rendering(path)
    rgb, fg = _render(path, load_mesh_albedo(path, V, F))
    assert fg.sum() > 100, "nothing rendered"
    px = rgb[fg].astype(np.float32)
    assert px[:, 0].mean() > px[:, 1].mean() + 40, (
        f"foreground is not red: mean RGB {px.mean(0)}")


def test_albedo_none_keeps_the_beige_fallback(tmp_path):
    """A mesh with no colour has nothing to show; beige is correct there."""
    path = _box(tmp_path)
    rgb, fg = _render(path, None)
    px = rgb[fg].astype(np.float32).mean(0)
    assert px[0] > px[1] > px[2], f"expected warm beige, got {px}"


def test_colour_actually_changes_the_image(tmp_path):
    """Guards against an albedo that is threaded through but ignored."""
    path = _box(tmp_path, color=(30, 30, 200))
    V, F, _, _, _ = load_mesh_for_rendering(path)
    coloured, fg = _render(path, load_mesh_albedo(path, V, F))
    beige, _ = _render(path, None)
    assert not np.array_equal(coloured[fg], beige[fg])
    px = coloured[fg].astype(np.float32)
    assert px[:, 2].mean() > px[:, 0].mean() + 40, "blue box did not render blue"
