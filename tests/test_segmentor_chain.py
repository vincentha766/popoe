"""The fallback contract: a chain routes around UNAVAILABLE segmentors, records
which one actually ran, and never hides a real failure.

Pure numpy — no GPU, no SAM2, no DINOv2.
"""

import numpy as np
import pytest

from popoe.interfaces import Detection, ObjectModel, Scene, Segmentor
from popoe.segmentor import (
    DepthSegmentor, FirstAvailableSegmentor, SegmentorUnavailable,
)


def _scene(depth=None):
    return Scene(rgb=np.zeros((32, 32, 3), np.uint8),
                 depth=np.zeros((32, 32), np.float32) if depth is None else depth,
                 K=np.eye(3), scene_id=1, im_id=2)


def _obj():
    return ObjectModel(obj_id=7, mesh_path="/nonexistent.ply", diameter=0.1)


class _Unavailable:
    source = "needs-a-checkpoint"

    def segment(self, scene, obj):
        raise SegmentorUnavailable("checkpoint not found")


class _Broken:
    source = "broken"

    def segment(self, scene, obj):
        raise RuntimeError("CUDA out of memory")


class _Works:
    source = "works"

    def __init__(self, n=2):
        self.n = n

    def segment(self, scene, obj):
        return [Detection(mask=np.ones((4, 4), bool), score=0.9)
                for _ in range(self.n)]


def test_chain_skips_unavailable_and_records_what_ran():
    chain = FirstAvailableSegmentor([_Unavailable(), _Works()])
    dets = chain.segment(_scene(), _obj())

    assert len(dets) == 2
    assert chain.last_used == "works"
    # every detection is stamped, so provenance survives into the CSV / cache key
    assert {d.source for d in dets} == {"works"}


def test_chain_prefers_the_first_available():
    chain = FirstAvailableSegmentor([_Works(n=1), _Works(n=3)])
    assert len(chain.segment(_scene(), _obj())) == 1
    assert chain.last_used == "works"


def test_runtime_failure_propagates_and_does_not_fall_back():
    """The old code caught bare Exception and fell back — an OOM looked like a
    successful (worse) segmentation. It must surface instead."""
    chain = FirstAvailableSegmentor([_Broken(), _Works()])
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        chain.segment(_scene(), _obj())


def test_empty_result_is_an_answer_not_a_failure():
    """No object in this image is legitimate; it must not silently advance to a
    weaker method (the old SAMSegmentor did exactly that)."""
    chain = FirstAvailableSegmentor([_Works(n=0), _Works(n=2)])
    assert chain.segment(_scene(), _obj()) == []
    assert chain.last_used == "works"

    opted_in = FirstAvailableSegmentor([_Works(n=0), _Works(n=2)],
                                       advance_on_empty=True)
    assert len(opted_in.segment(_scene(), _obj())) == 2


def test_exhausted_chain_raises_rather_than_returning_nothing():
    chain = FirstAvailableSegmentor([_Unavailable()])
    with pytest.raises(SegmentorUnavailable, match="no segmentor in the chain"):
        chain.segment(_scene(), _obj())
    assert chain.last_used is None


def test_depth_segmentor_runs_with_no_deps_and_scores_by_area():
    pytest.importorskip("cv2")   # the only heavy dep below the chain layer
    depth = np.zeros((32, 32), np.float32)
    depth[4:20, 4:20] = 1.0      # big blob: 256 px
    depth[24:30, 24:30] = 1.05   # small blob: 36 px
    dets = DepthSegmentor(min_pixels=10, kernel=3).segment(_scene(depth), _obj())

    assert len(dets) == 2
    assert dets[0].score > dets[1].score            # biggest blob first
    assert dets[0].mask.sum() > dets[1].mask.sum()
    assert {d.source for d in dets} == {"depth-cc"}
    # score IS the area fraction — not a confidence, and not comparable to a
    # DINO cosine similarity. Documented so nobody sorts the two together again.
    h, w = depth.shape
    assert dets[0].score == pytest.approx(dets[0].mask.sum() / (h * w))


def test_segmentors_satisfy_the_stage_protocol():
    assert isinstance(DepthSegmentor(), Segmentor)
    assert isinstance(FirstAvailableSegmentor([DepthSegmentor()]), Segmentor)
