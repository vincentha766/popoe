"""Unit tests for render_rerank pure helpers (no GPU)."""
import numpy as np
import pytest

from popoe.render_rerank import (
    bbox_from_mask,
    pca_flip_variants,
    pick_by_scores,
    rot_about,
    RenderAppearanceReranker,
)
from popoe.interfaces import PoseHypothesis, PointFeatures, Scene, ObjectModel


def test_rot_about_180_is_involutory():
    R = rot_about([0, 0, 1], 180)
    assert np.allclose(R @ R, np.eye(3), atol=1e-9)


def test_pca_flip_variants_set():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 3))
    pts[:, 2] *= 3  # elongated
    R0 = np.eye(3)
    vs = pca_flip_variants(R0, pts, include_azimuth=True)
    names = [n for n, _ in vs]
    assert names[0] == "champion"
    assert "flip0" in names and "flip1" in names and "flip2" in names
    assert "az90" in names and "az270" in names
    assert len(vs) == 6  # champion + 3 flips + 2 az


def test_pca_flip_variants_no_azimuth():
    pts = np.eye(3)
    vs = pca_flip_variants(np.eye(3), pts, include_azimuth=False)
    assert len(vs) == 4  # champion + 3 flips


def test_pick_by_scores_argmax():
    R0, R1 = np.eye(3), rot_about([0, 1, 0], 180)
    vs = [("champion", R0), ("flip0", R1)]
    name, R, s = pick_by_scores(vs, {"champion": 0.1, "flip0": 0.9})
    assert name == "flip0"
    assert s == 0.9
    assert np.allclose(R, R1)


def test_bbox_from_mask():
    m = np.zeros((10, 12), dtype=bool)
    m[2:5, 3:7] = True
    assert bbox_from_mask(m) == (3, 2, 7, 5)
    assert bbox_from_mask(np.zeros((4, 4), dtype=bool)) is None


def test_reranker_disabled_is_identity():
    rr = RenderAppearanceReranker(enabled=False)
    pose = PoseHypothesis(np.eye(3), np.zeros(3), 0.5, {"k": 1})
    scene = Scene(rgb=np.zeros((8, 8, 3), np.uint8),
                  depth=np.zeros((8, 8), np.float32),
                  K=np.eye(3))
    obj = ObjectModel(obj_id=1, mesh_path="/tmp/x.ply", diameter=0.1)
    q = PointFeatures(pts=np.eye(3), feats=np.eye(3))
    t = PointFeatures(pts=np.eye(3), feats=np.eye(3))
    out = rr.refine(pose, scene, obj, q, t)
    assert out.score == 0.5
    assert np.allclose(out.R, pose.R)


def test_reranker_skips_without_bbox():
    rr = RenderAppearanceReranker(enabled=True)
    # Force enabled path but no bbox → skip without loading GPU backends.
    pose = PoseHypothesis(np.eye(3), np.zeros(3), 0.42, {})
    scene = Scene(rgb=np.zeros((8, 8, 3), np.uint8),
                  depth=np.zeros((8, 8), np.float32),
                  K=np.eye(3))
    obj = ObjectModel(obj_id=1, mesh_path="/tmp/x.ply", diameter=0.1)
    q = PointFeatures(pts=np.random.randn(20, 3), feats=np.random.randn(20, 4))
    t = PointFeatures(pts=np.random.randn(20, 3), feats=np.random.randn(20, 4),
                      meta={})  # no detection / bbox
    out = rr.refine(pose, scene, obj, q, t)
    assert out.breakdown.get("render_rerank") == "skipped_no_bbox"
    assert out.score == 0.42
