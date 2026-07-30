"""Target-encoding regressions for degenerate masks and GeDi batch edges."""

import numpy as np
import pytest

pytest.importorskip("torch")

from popoe.freeze.adapters import FreeZeTargetEncoder
from popoe.freeze.feature_extractor import (
    TargetFeatureExtractor,
    _NoSingletonGeDiBatches,
)
from popoe.interfaces import CanonFrame, Detection, ObjectModel, Scene


class _SingletonBatchFragileDescriptor:
    def __init__(self, samples_per_batch=500):
        self.samples_per_batch = samples_per_batch
        self.calls = []

    def compute(self, pts, pcd):
        n = int(pts.shape[0])
        self.calls.append(n)
        for i in range(0, n, self.samples_per_batch):
            if int(pts[i:i + self.samples_per_batch].shape[0]) == 1:
                raise RuntimeError(
                    "linalg.cross: inputs must have the same number of dimensions.")
        arr = pts.cpu().numpy() if hasattr(pts, "cpu") else np.asarray(pts)
        return arr[:, :1].astype(np.float32)


def test_gedi_singleton_final_batch_is_padded_and_trimmed():
    pts = np.arange(501 * 3, dtype=np.float32).reshape(501, 3)
    raw = _SingletonBatchFragileDescriptor()
    with pytest.raises(RuntimeError, match="linalg.cross"):
        raw.compute(pts, pts)

    wrapped = _NoSingletonGeDiBatches(
        _SingletonBatchFragileDescriptor(), samples_per_batch=500)
    got = wrapped.compute(pts, pts)

    assert wrapped.descriptor.calls == [502]
    assert got.shape == (501, 1)
    np.testing.assert_array_equal(got[:, 0], pts[:, 0])


def test_gedi_non_singleton_batches_are_unchanged():
    pts = np.arange(500 * 3, dtype=np.float32).reshape(500, 3)
    raw = _SingletonBatchFragileDescriptor()
    want = raw.compute(pts, pts)

    wrapped = _NoSingletonGeDiBatches(
        _SingletonBatchFragileDescriptor(), samples_per_batch=500)
    got = wrapped.compute(pts, pts)

    assert wrapped.descriptor.calls == [500]
    np.testing.assert_array_equal(got, want)


class _GeoDescriptor:
    def compute(self, pts, pcd):
        n = int(pts.shape[0])
        return np.ones((n, 2), dtype=np.float32)


class _TargetExtractor(TargetFeatureExtractor):
    def _extract_dino_at_points(self, rgb, us, vs):
        return np.stack([us, vs], axis=1).astype(np.float32)

    def _fuse_features(self, vis_feats, geo_feats):
        return np.concatenate([vis_feats, geo_feats], axis=1).astype(np.float32)


def _target_encoder():
    return FreeZeTargetEncoder(
        _TargetExtractor(device="cpu", dino=object(), gedi=_GeoDescriptor()))


def _scene(depth):
    K = np.array([[100.0, 0.0, 0.0],
                  [0.0, 100.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return Scene(rgb=np.zeros(depth.shape + (3,), np.uint8), depth=depth, K=K)


def _obj():
    return ObjectModel(obj_id=1, mesh_path="obj_000001.ply", diameter=0.1)


def test_empty_mask_degrades_with_reason():
    depth = np.ones((8, 8), dtype=np.float32)
    det = Detection(mask=np.zeros((8, 8), dtype=bool), score=1.0)

    got = _target_encoder().encode_target(
        _scene(depth), det, _obj(), CanonFrame(np.zeros(3), 1.0))

    assert got.pts.shape == (0, 3)
    assert got.meta["skip_reason"] == "empty mask"


def test_duplicate_grid_points_do_not_hide_tiny_depth_support():
    depth = np.zeros((8, 8), dtype=np.float32)
    depth[3, 3] = 1.0
    mask = np.zeros((8, 8), dtype=bool)
    mask[3, 3] = True
    det = Detection(mask=mask, score=1.0)

    got = _target_encoder().encode_target(
        _scene(depth), det, _obj(), CanonFrame(np.zeros(3), 1.0))

    assert got.pts.shape == (0, 3)
    assert got.meta["skip_reason"] == "only 1 valid depth pixel(s) in mask"


def test_grid_depth_holes_resample_from_valid_mask_pixels(monkeypatch):
    monkeypatch.setenv("POPOE_TARGET_GRID", "2")
    depth = np.zeros((8, 8), dtype=np.float32)
    depth[3:5, 3:5] = 1.0
    mask = np.ones((8, 8), dtype=bool)
    det = Detection(mask=mask, score=1.0)

    got = _target_encoder().encode_target(
        _scene(depth), det, _obj(), CanonFrame(np.zeros(3), 1.0))

    assert got.pts.shape == (4, 3)
    assert "skip_reason" not in got.meta


def test_healthy_target_still_encodes(monkeypatch):
    monkeypatch.setenv("POPOE_TARGET_GRID", "4")
    depth = np.zeros((8, 8), dtype=np.float32)
    depth[2:6, 2:6] = 1.0
    mask = depth > 0
    det = Detection(mask=mask, score=1.0)

    got = _target_encoder().encode_target(
        _scene(depth), det, _obj(), CanonFrame(np.zeros(3), 1.0))

    assert got.pts.shape == (16, 3)
    assert got.feats.shape == (16, 4)
    assert "skip_reason" not in got.meta

    ys, xs = np.meshgrid(np.arange(2, 6), np.arange(2, 6), indexing="ij")
    us = xs.flatten()
    vs = ys.flatten()
    want_pts = np.stack([us / 100.0, vs / 100.0,
                         np.ones_like(us, dtype=np.float32)], axis=1)
    want_feats = np.concatenate([
        np.stack([us, vs], axis=1).astype(np.float32),
        np.ones((16, 2), dtype=np.float32),
    ], axis=1)
    np.testing.assert_allclose(got.pts, want_pts, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(got.feats, want_feats)
