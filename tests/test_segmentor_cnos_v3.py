import numpy as np
import pytest

from popoe.interfaces import ObjectModel, Scene
from popoe.segmentor import SegmentorUnavailable
from popoe.segmentor_cnos_v3 import (
    CNOSv3Segmentor,
    DepthSizeGate,
    PatchForegroundScorer,
)


def _scene():
    K = np.array([[100.0, 0.0, 20.0],
                  [0.0, 100.0, 20.0],
                  [0.0, 0.0, 1.0]])
    return Scene(rgb=np.zeros((40, 40, 3), np.uint8),
                 depth=np.ones((40, 40), np.float32),
                 K=K, scene_id=0, im_id=1)


def _mask(y0=10, y1=30, x0=10, x1=30):
    m = np.zeros((40, 40), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_depth_size_gate_keeps_plausible_extent_and_rejects_wrong_size():
    scene = _scene()
    gate = DepthSizeGate(min_pixels=1, min_points=10)
    mask = _mask()
    extent = gate.extent_3d(mask, scene.depth, scene.K)

    ok, same_extent = gate.accepts(scene, ObjectModel(1, "x", diameter=0.25), mask)
    too_small, _ = gate.accepts(scene, ObjectModel(1, "x", diameter=0.05), mask)

    assert extent is not None
    assert same_extent == pytest.approx(extent)
    assert ok is True
    assert too_small is False


def test_patch_foreground_scorer_uses_foreground_and_topk():
    scorer = PatchForegroundScorer(topk=2)
    fg = np.zeros((224, 224), bool)
    fg[:14, :14] = True
    keep = scorer.foreground_patch_mask(fg)
    assert keep.shape == (16 * 16,)
    assert keep.sum() >= 3  # tiny fg falls back to all patches

    q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    t = np.array([[1.0, 0.0]])
    assert scorer.score_tokens(q, t) == pytest.approx(1.0)


class _Proposer:
    def __init__(self, masks):
        self._masks = masks

    def propose(self, scene):
        return self._masks


class _Bank:
    def patches_for(self, obj):
        return np.array([[1.0, 0.0]])


class _Gate:
    def accepts(self, scene, obj, mask):
        return True, float(mask.sum())


class _Scorer:
    def score_mask(self, rgb, mask, template_tokens):
        return float(mask.sum())


def test_cnos_v3_segmentor_sorts_and_stamps_source():
    small = _mask(1, 8, 1, 8)
    large = _mask(10, 30, 10, 30)
    seg = CNOSv3Segmentor(
        proposer=_Proposer([small, large]),
        template_bank=_Bank(),
        scorer=_Scorer(),
        size_gate=_Gate(),
        n_masks=1,
    )

    dets = seg.segment(_scene(), ObjectModel(9, "x", diameter=0.25))

    assert len(dets) == 1
    assert dets[0].source == "cnos-v3"
    assert dets[0].score == pytest.approx(float(large.sum()))
    assert dets[0].bbox == (10.0, 10.0, 30.0, 30.0)


def test_cnos_v3_requires_explicit_heavy_components():
    with pytest.raises(SegmentorUnavailable, match="mask proposer"):
        CNOSv3Segmentor().segment(_scene(), ObjectModel(9, "x", diameter=0.25))
