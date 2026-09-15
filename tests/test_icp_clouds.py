"""Method-agnostic ICP clouds: metres, mask crop, mm CAD conversion."""
import numpy as np
import pytest

from popoe.adapters import cad_surface_cloud, depth_mask_cloud, icp_clouds
from popoe.interfaces import Detection, ObjectModel, Scene


def _scene(depth, K=None, mask_hw=None):
    K = (np.array([[500., 0., 32.], [0., 500., 24.], [0., 0., 1.]])
         if K is None else K)
    rgb = np.zeros(depth.shape + (3,), np.uint8)
    return Scene(rgb=rgb, depth=depth, K=K, scene_id=1, im_id=1)


def test_depth_mask_cloud_metres_and_drops_holes():
    depth = np.zeros((48, 64), np.float32)
    depth[10:14, 20:25] = 0.8
    depth[12, 22] = 0.0
    mask = np.zeros((48, 64), bool)
    mask[10:14, 20:25] = True
    got = depth_mask_cloud(_scene(depth), mask, max_pts=0)
    ys, xs = np.where((depth > 0) & mask)
    d = depth[ys, xs]
    want = np.stack([(xs - 32.) * d / 500., (ys - 24.) * d / 500., d], 1)
    assert got.shape == (19, 3)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)


def test_depth_mask_cloud_without_mask_uses_all_valid_depth():
    depth = np.zeros((8, 8), np.float32)
    depth[2:6, 2:6] = 1.0
    got = depth_mask_cloud(_scene(depth), mask=None, max_pts=0)
    assert len(got) == 16
    assert np.allclose(got[:, 2], 1.0)


def test_depth_mask_cloud_cap_and_degenerate():
    depth = np.full((48, 64), 0.9, np.float32)
    mask = np.ones((48, 64), bool)
    sc = _scene(depth)
    a = depth_mask_cloud(sc, mask, max_pts=100)
    b = depth_mask_cloud(sc, mask, max_pts=100)
    assert len(a) == 100
    np.testing.assert_array_equal(a, b)
    tiny_depth = np.zeros((8, 8), np.float32)
    tiny_depth[1, 1] = 0.5
    tiny = np.zeros((8, 8), bool)
    tiny[1, 1] = True
    assert depth_mask_cloud(_scene(tiny_depth), tiny, max_pts=0) is None


def test_cad_surface_cloud_converts_mm_when_extent_dwarfs_diameter(monkeypatch):
    def fake_sample(path, n, seed):
        assert path == "obj.ply" and n == 8 and seed == 9
        return np.array([[0., 0., 0.], [200., 0., 0.]], dtype=np.float64)

    monkeypatch.setattr("popoe.freeze.adapters.sample_query_surface",
                        fake_sample)
    obj = ObjectModel(9, "obj.ply", diameter=0.1)
    pts = cad_surface_cloud(obj, n_points=8, seed=9)
    np.testing.assert_allclose(pts[-1], [0.2, 0, 0])


def test_cad_surface_cloud_keeps_metre_meshes(monkeypatch):
    def fake_sample(path, n, seed):
        return np.array([[0., 0., 0.], [0.08, 0., 0.]])

    monkeypatch.setattr("popoe.freeze.adapters.sample_query_surface",
                        fake_sample)
    obj = ObjectModel(1, "obj.ply", diameter=0.1)
    pts = cad_surface_cloud(obj, n_points=2, seed=1)
    np.testing.assert_allclose(pts[-1], [0.08, 0, 0])


def test_icp_clouds_uses_detection_mask(monkeypatch):
    monkeypatch.setattr(
        "popoe.adapters.cad_surface_cloud",
        lambda obj, n_points=3000, seed=None: np.zeros((4, 3), np.float32))
    depth = np.zeros((8, 8), np.float32)
    depth[2:6, 2:6] = 0.5
    mask = np.zeros((8, 8), bool)
    mask[2:4, 2:4] = True
    scene = _scene(depth)
    obj = ObjectModel(1, "obj.ply", 0.1)
    src, tgt = icp_clouds(scene, obj, Detection(mask=mask, score=1.0),
                          max_scene=0)
    assert src.shape == (4, 3)
    assert len(tgt) == 4
    with pytest.raises(ValueError, match="fewer than 4"):
        icp_clouds(scene, obj, Detection(mask=np.zeros((8, 8), bool), score=0))
