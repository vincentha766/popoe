import json

import numpy as np

from popoe.interfaces import ObjectModel, Scene
from popoe.segmentor_cnos_official import (
    CNOSDetectionsSegmentor,
    build_cnos_bop_command,
    build_cnos_custom_infer_command,
    build_cnos_custom_render_command,
)


def _scene():
    return Scene(rgb=np.zeros((8, 8, 3), np.uint8),
                 depth=np.zeros((8, 8), np.float32),
                 K=np.eye(3), scene_id=0, im_id=42)


def test_cnos_detections_segmentor_stamps_official_source(tmp_path):
    mask = np.zeros((8, 8), np.uint8)
    mask[2:6, 1:5] = 1
    np.save(tmp_path / "mask.npy", mask)
    p = tmp_path / "cnos.json"
    p.write_text(json.dumps([{
        "scene_id": 0,
        "image_id": 42,
        "category_id": 9,
        "score": 0.91,
        "bbox": [1, 2, 4, 4],
        "mask_path": "mask.npy",
    }]))

    seg = CNOSDetectionsSegmentor(str(p), topk=1, min_pixels=1)
    dets = seg.segment(_scene(), ObjectModel(obj_id=9, mesh_path="x", diameter=0.1))

    assert len(dets) == 1
    assert dets[0].source == "cnos"
    assert dets[0].bbox == (1.0, 2.0, 5.0, 6.0)
    assert np.array_equal(dets[0].mask, mask.astype(bool))


def test_cnos_bop_command_builder():
    cmd = build_cnos_bop_command(
        "lmo",
        cnos_root="/repo/cnos",
        python="/env/bin/python",
        model="fastsam",
        rendering_type="pyrender",
        level_templates=1,
        gpu="0",
        overrides=["hydra.run.dir=/tmp/cnos"],
    )

    assert cmd.cwd == "/repo/cnos"
    assert list(cmd.argv) == [
        "/env/bin/python",
        "run_inference.py",
        "dataset_name=lmo",
        "model=cnos_fast",
        "model.onboarding_config.rendering_type=pyrender",
        "model.onboarding_config.level_templates=1",
        "hydra.run.dir=/tmp/cnos",
    ]
    assert dict(cmd.env) == {"CUDA_VISIBLE_DEVICES": "0"}


def test_cnos_custom_command_builders():
    render = build_cnos_custom_render_command(
        "/cad/obj.ply", "/out", cnos_root="/repo/cnos", rgb_path="/rgb.png")
    assert render.cwd == "/repo/cnos"
    assert list(render.argv) == ["bash", "./src/scripts/render_custom.sh"]
    assert dict(render.env) == {
        "CAD_PATH": "/cad/obj.ply",
        "OUTPUT_DIR": "/out",
        "RGB_PATH": "/rgb.png",
    }

    infer = build_cnos_custom_infer_command(
        "/rgb.png",
        "/out",
        cnos_root="/repo/cnos",
        python="/env/bin/python",
        num_max_dets=3,
        conf_threshold=0.4,
        stability_score_thresh=0.6,
    )
    assert infer.cwd == "/repo/cnos"
    assert list(infer.argv) == [
        "/env/bin/python",
        "-m",
        "src.scripts.inference_custom",
        "--template_dir",
        "/out",
        "--rgb_path",
        "/rgb.png",
        "--num_max_dets",
        "3",
        "--confg_threshold",
        "0.4",
        "--stability_score_thresh",
        "0.6",
    ]
