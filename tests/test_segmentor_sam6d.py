import json
import warnings

import numpy as np

from popoe.interfaces import ObjectModel, Scene
from popoe.segmentor_sam6d import (
    SAM6DIsmDetectionsSegmentor,
    SAM6DPemResultsCoarseEstimator,
    build_ism_bop_command,
    build_pem_bop_command,
    load_sam6d_pem_csv,
    load_sam6d_pem_json,
)


def _scene():
    return Scene(rgb=np.zeros((8, 8, 3), np.uint8),
                 depth=np.zeros((8, 8), np.float32),
                 K=np.eye(3), scene_id=0, im_id=42)


def _obj():
    return ObjectModel(obj_id=9, mesh_path="obj.ply", diameter=0.1)


def test_sam6d_ism_segmentor_reads_detection_file(tmp_path):
    mask = np.zeros((8, 8), np.uint8)
    mask[1:5, 2:6] = 1
    np.save(tmp_path / "mask.npy", mask)
    det_path = tmp_path / "sam6d_ism.json"
    det_path.write_text(json.dumps([{
        "scene_id": 0,
        "image_id": 42,
        "category_id": 9,
        "score": 0.93,
        "bbox": [2, 1, 4, 4],
        "mask_path": "mask.npy",
    }]))

    seg = SAM6DIsmDetectionsSegmentor(str(det_path), topk=1, min_pixels=1)
    dets = seg.segment(_scene(), _obj())

    assert len(dets) == 1
    assert dets[0].source == "sam6d"
    assert dets[0].bbox == (2.0, 1.0, 6.0, 5.0)
    assert np.array_equal(dets[0].mask, mask.astype(bool))


def test_sam6d_command_builders_use_official_working_dirs(tmp_path):
    root = tmp_path / "SAM-6D"
    ism = build_ism_bop_command(
        "lmo",
        sam6d_root=str(root),
        python="/envs/sam6d/bin/python",
        model="fastsam",
        gpu="0",
        overrides=["hydra.run.dir=/tmp/ism"],
    )
    assert ism.cwd == str(root / "SAM-6D" / "Instance_Segmentation_Model")
    assert list(ism.argv) == [
        "/envs/sam6d/bin/python",
        "run_inference.py",
        "dataset_name=lmo",
        "model=ISM_fastsam",
        "hydra.run.dir=/tmp/ism",
    ]
    assert dict(ism.env) == {"CUDA_VISIBLE_DEVICES": "0"}

    pem = build_pem_bop_command(
        "ycbv",
        sam6d_root=str(root),
        python="/envs/sam6d/bin/python",
        checkpoint_path="checkpoints/sam-6d-pem-base.pth",
        gpus="0",
        view=42,
    )
    assert pem.cwd == str(root / "SAM-6D" / "Pose_Estimation_Model")
    assert list(pem.argv) == [
        "/envs/sam6d/bin/python",
        "test_bop.py",
        "--gpus",
        "0",
        "--model",
        "pose_estimation_model",
        "--config",
        "config/base.yaml",
        "--dataset",
        "ycbv",
        "--checkpoint_path",
        "checkpoints/sam-6d-pem-base.pth",
        "--view",
        "42",
    ]


def test_sam6d_pem_csv_loader_and_coarse_estimator(tmp_path):
    csv_path = tmp_path / "result_lmo.csv"
    csv_path.write_text(
        "scene_id,im_id,obj_id,score,R,t,time\n"
        "0,42,9,0.25,1 0 0 0 1 0 0 0 1,100 200 300,0.50\n"
        "0,42,9,0.75,0 -1 0 1 0 0 0 0 1,10 20 30,0.25\n"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        preds = load_sam6d_pem_csv(str(csv_path))
    assert caught == []
    assert len(preds) == 2
    assert np.allclose(preds[1].hypothesis.t, [0.01, 0.02, 0.03])

    est = SAM6DPemResultsCoarseEstimator(str(csv_path), topk=1)
    hyps = est.estimate(_scene(), _obj())
    assert len(hyps) == 1
    assert hyps[0].score == 0.75
    assert hyps[0].breakdown["source"] == "sam6d-pem"


def test_sam6d_pem_json_loader_supports_custom_demo_output(tmp_path):
    json_path = tmp_path / "detection_pem.json"
    json_path.write_text(json.dumps([{
        "scene_id": 999,
        "image_id": 999,
        "category_id": 0,
        "score": 0.8,
        "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "t": [1, 2, 3],
    }]))

    preds = load_sam6d_pem_json(str(json_path), scene_id=0, image_id=42, obj_id=9)

    assert len(preds) == 1
    assert (preds[0].scene_id, preds[0].im_id) == (0, 42)
    assert preds[0].obj_id == 9
    assert np.allclose(preds[0].hypothesis.t, [0.001, 0.002, 0.003])
