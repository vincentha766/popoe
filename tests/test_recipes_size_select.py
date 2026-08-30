"""Recipe wiring for opt-in YCB-V clamp size_select (no GPU)."""

import json

import numpy as np
import pytest

from popoe.freeze.recipes import (
    YCBV_CLAMP_DIAMETERS_M,
    YCBV_MERGE_LABELS,
    best_segmentor,
)
from popoe.interfaces import ObjectModel, Scene


def _rle(mask):
    from pycocotools import mask as cm
    r = cm.encode(np.asfortranarray(mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode()
    return {"size": list(mask.shape), "counts": r["counts"]}


def test_best_segmentor_can_set_nearest_and_merge(tmp_path):
    H, W = 32, 32
    m = np.zeros((H, W), bool)
    m[8:24, 8:24] = True
    dets = [{
        "scene_id": 1, "image_id": 1, "category_id": 19,
        "score": 0.5, "segmentation": _rle(m),
    }]
    p = tmp_path / "d.json"
    p.write_text(json.dumps(dets))

    seg = best_segmentor(
        str(p), topk=1, merge_labels=YCBV_MERGE_LABELS,
        size_select="nearest",
        confusable_diameters=dict(YCBV_CLAMP_DIAMETERS_M),
    )
    assert seg.merge_labels == YCBV_MERGE_LABELS
    assert seg.size_select == "nearest"
    assert seg.confusable_diameters == YCBV_CLAMP_DIAMETERS_M

    plain = best_segmentor(str(p), topk=1, merge_labels=YCBV_MERGE_LABELS)
    assert plain.size_select is None


def test_best_segmentor_size_select_soft_without_depth(tmp_path):
    H, W = 32, 32
    m = np.zeros((H, W), bool)
    m[4:28, 4:28] = True
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{
        "scene_id": 1, "image_id": 1, "category_id": 19,
        "score": 0.9, "segmentation": _rle(m),
    }]))
    seg = best_segmentor(
        str(p), topk=1, merge_labels=YCBV_MERGE_LABELS,
        size_select="soft", confusable_diameters=YCBV_CLAMP_DIAMETERS_M,
    )
    assert seg.size_select == "soft"
    # depth-less scene: size_select is a no-op and still returns dets
    scene = Scene(
        rgb=np.zeros((H, W, 3), np.uint8),
        depth=None, K=None, scene_id=1, im_id=1,
    )
    out = seg.segment(scene, ObjectModel(19, "x", diameter=0.17))
    assert len(out) == 1
