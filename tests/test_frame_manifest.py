import json

import numpy as np

from popoe.datasets.frames import load_frame_manifest, load_scene_from_manifest
from popoe.interfaces import ObjectModel
from popoe.segmentor_detections import BOPDetectionsSegmentor


def test_real_scene_manifest_loads_scene_and_2d_mask_detections(tmp_path):
    rgb = np.zeros((3, 4, 3), np.uint8)
    rgb[..., 0] = 17
    depth_raw = np.array([[0, 1000, 1001, 0],
                          [0, 2000, 2001, 0],
                          [0, 3000, 3001, 0]], dtype=np.uint16)
    mask = np.zeros((3, 4), np.uint8)
    mask[1:3, 2:4] = 1

    np.save(tmp_path / "rgb.npy", rgb)
    np.save(tmp_path / "depth.npy", depth_raw)
    np.save(tmp_path / "mask.npy", mask)

    dets_path = tmp_path / "detections.json"
    dets_path.write_text(json.dumps([{
        "scene_id": 0,
        "image_id": 42,
        "category_id": 9,
        "score": 0.96,
        "bbox": [2, 1, 2, 2],
        "mask": "mask.npy",
    }]))
    manifest_path = tmp_path / "frame.json"
    manifest_path.write_text(json.dumps({
        "scene_id": 0,
        "image_id": 42,
        "rgb_path": "rgb.npy",
        "depth_path": "depth.npy",
        "depth_scale": 0.001,
        "K": [[100.0, 0.0, 2.0], [0.0, 101.0, 1.5], [0.0, 0.0, 1.0]],
        "detections_path": "detections.json",
        "capture_id": "bench-001",
    }))

    manifest = load_frame_manifest(manifest_path)
    assert manifest.im_id == 42
    assert manifest.meta["capture_id"] == "bench-001"
    scene = load_scene_from_manifest(manifest)
    assert scene.scene_id == 0 and scene.im_id == 42
    assert scene.depth.dtype == np.float32
    assert np.isclose(scene.depth[1, 2], np.float32(2.001))
    assert np.array_equal(scene.rgb, rgb)

    seg = BOPDetectionsSegmentor(manifest.detections_path, source="local",
                                 topk=1, min_pixels=1)
    out = seg.segment(scene, ObjectModel(obj_id=9, mesh_path="obj.ply",
                                         diameter=0.1))
    assert len(out) == 1
    assert out[0].source == "local"
    assert out[0].bbox == (2.0, 1.0, 4.0, 3.0)
    assert np.array_equal(out[0].mask, mask.astype(bool))
