import json

import numpy as np
import pytest

from popoe.interfaces import ObjectModel, Scene
from popoe.segmentor_nids import (
    NIDSNetDetectionsSegmentor,
    adapt_nidsnet_json,
    adapt_nidsnet_records,
)


def _scene():
    return Scene(rgb=np.zeros((8, 8, 3), np.uint8),
                 depth=np.zeros((8, 8), np.float32),
                 K=np.eye(3), scene_id=0, im_id=42)


def test_adapt_nidsnet_records_stamps_ids_and_maps_category():
    raw = [{
        "image_id": 999,
        "category_id": 1,
        "score": "0.91",
        "bbox": [1, 2, 3, 4],
        "mask_path": "mask.npy",
    }]
    out = adapt_nidsnet_records(
        raw, scene_id=0, image_id=42, category_id_map={1: 9})
    assert out[0]["scene_id"] == 0
    assert out[0]["image_id"] == 42
    assert out[0]["category_id"] == 9
    assert out[0]["score"] == pytest.approx(0.91)
    assert out[0]["source"] == "nids"


def test_adapt_nidsnet_records_rejects_maskless_output():
    with pytest.raises(ValueError, match="no mask"):
        adapt_nidsnet_records([{"image_id": 1, "category_id": 9}])


def test_nids_segmentor_reads_adapted_mask_path_output(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "converted"
    raw_dir.mkdir()
    mask = np.zeros((8, 8), np.uint8)
    mask[2:6, 1:5] = 1
    np.save(raw_dir / "mask.npy", mask)

    raw_path = raw_dir / "nids_raw.json"
    raw_path.write_text(json.dumps([{
        "image_id": 42,
        "category_id": 1,
        "score": 0.88,
        "bbox": [1, 2, 4, 4],
        "mask_path": "mask.npy",
    }]))
    out_path = out_dir / "detections.json"
    records = adapt_nidsnet_json(
        str(raw_path), str(out_path), scene_id=0, image_id=42,
        category_id_map={1: 9})
    assert len(records) == 1
    assert records[0]["mask_path"] == str(raw_dir / "mask.npy")

    seg = NIDSNetDetectionsSegmentor(str(out_path), topk=1, min_pixels=1)
    dets = seg.segment(_scene(), ObjectModel(obj_id=9, mesh_path="x",
                                             diameter=0.1))
    assert len(dets) == 1
    assert dets[0].source == "nids"
    assert dets[0].bbox == (1.0, 2.0, 5.0, 6.0)
    assert np.array_equal(dets[0].mask, mask.astype(bool))
