import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")
pytest.importorskip("pycocotools")

from PIL import Image

from popoe.segmentation_eval import (
    bop_image_id,
    build_coco_gt_from_bop,
    detections_to_coco_results,
    evaluate_coco_segm,
    filter_coco_gt,
    load_bop_targets,
    write_json,
)


def _rle(mask):
    from pycocotools import mask as coco_mask

    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return {"size": list(mask.shape), "counts": rle["counts"]}


def _mask(height=32, width=40, y0=6, x0=10, h=12, w=14):
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y0 + h, x0:x0 + w] = True
    return mask


def _write_bop_scene(root: Path, mask: np.ndarray, *, obj_id=5):
    scene = root / "test" / "000001"
    (scene / "mask_visib").mkdir(parents=True)
    (scene / "rgb").mkdir()
    Image.fromarray(np.zeros((*mask.shape, 3), dtype=np.uint8)).save(
        scene / "rgb" / "000007.png")
    Image.fromarray((mask.astype(np.uint8) * 255)).save(
        scene / "mask_visib" / "000007_000000.png")
    (scene / "scene_gt.json").write_text(json.dumps({
        "7": [{"obj_id": obj_id}]
    }))
    return scene


def _write_detections(path: Path, mask: np.ndarray, *, obj_id=5, score=0.99):
    path.write_text(json.dumps([{
        "scene_id": 1,
        "image_id": 7,
        "category_id": obj_id,
        "score": score,
        "segmentation": _rle(mask),
    }]))


def test_build_coco_gt_from_bop_visible_masks(tmp_path):
    gt_mask = _mask()
    _write_bop_scene(tmp_path, gt_mask)

    coco_gt = build_coco_gt_from_bop(tmp_path)

    assert coco_gt["images"][0]["id"] == bop_image_id(1, 7)
    assert coco_gt["images"][0]["width"] == gt_mask.shape[1]
    assert coco_gt["annotations"][0]["category_id"] == 5
    assert coco_gt["annotations"][0]["bbox"] == [10.0, 6.0, 14.0, 12.0]
    assert coco_gt["categories"] == [{"id": 5, "name": "obj_000005"}]


def test_convert_and_evaluate_perfect_segmentation(tmp_path):
    gt_mask = _mask()
    _write_bop_scene(tmp_path, gt_mask)
    det_path = tmp_path / "detections.json"
    _write_detections(det_path, gt_mask)

    coco_gt = build_coco_gt_from_bop(tmp_path)
    pred = detections_to_coco_results(det_path, coco_gt)

    assert len(pred) == 1
    assert pred[0]["image_id"] == bop_image_id(1, 7)
    assert pred[0]["category_id"] == 5

    gt_json = tmp_path / "gt_coco.json"
    pred_json = tmp_path / "pred_coco.json"
    write_json(gt_json, coco_gt)
    write_json(pred_json, pred)
    stats = evaluate_coco_segm(gt_json, pred_json)

    assert stats["AP"] == pytest.approx(1.0)
    assert stats["AP50"] == pytest.approx(1.0)


def test_targets_filter_gt_and_predictions(tmp_path):
    gt_mask = _mask()
    _write_bop_scene(tmp_path, gt_mask, obj_id=5)
    targets_path = tmp_path / "targets.json"
    targets_path.write_text(json.dumps([
        {"scene_id": 1, "im_id": 7, "obj_id": 5, "inst_count": 1},
        {"scene_id": 1, "im_id": 7, "obj_id": 9, "inst_count": 1},
    ]))
    det_path = tmp_path / "detections.json"
    det_path.write_text(json.dumps([
        {
            "scene_id": 1,
            "image_id": 7,
            "category_id": 5,
            "score": 0.9,
            "segmentation": _rle(gt_mask),
        },
        {
            "scene_id": 1,
            "image_id": 7,
            "category_id": 9,
            "score": 0.8,
            "segmentation": _rle(gt_mask),
        },
    ]))

    targets = {t for t in load_bop_targets(targets_path) if t[2] == 5}
    coco_gt = build_coco_gt_from_bop(tmp_path, targets=targets)
    pred = detections_to_coco_results(det_path, coco_gt, targets=targets)

    assert [ann["category_id"] for ann in coco_gt["annotations"]] == [5]
    assert [rec["category_id"] for rec in pred] == [5]


def test_category_filter_gt_and_predictions(tmp_path):
    mask5 = _mask(y0=4, x0=5)
    mask9 = _mask(y0=15, x0=20)
    scene = _write_bop_scene(tmp_path, mask5, obj_id=5)
    Image.fromarray((mask9.astype(np.uint8) * 255)).save(
        scene / "mask_visib" / "000007_000001.png")
    (scene / "scene_gt.json").write_text(json.dumps({
        "7": [{"obj_id": 5}, {"obj_id": 9}]
    }))
    det_path = tmp_path / "detections.json"
    det_path.write_text(json.dumps([
        {
            "scene_id": 1,
            "image_id": 7,
            "category_id": 5,
            "score": 0.9,
            "segmentation": _rle(mask5),
        },
        {
            "scene_id": 1,
            "image_id": 7,
            "category_id": 9,
            "score": 0.8,
            "segmentation": _rle(mask9),
        },
    ]))

    coco_gt = build_coco_gt_from_bop(tmp_path, category_ids={9})
    pred = detections_to_coco_results(det_path, coco_gt, category_ids={9})

    assert [ann["category_id"] for ann in coco_gt["annotations"]] == [9]
    assert [cat["id"] for cat in coco_gt["categories"]] == [9]
    assert [rec["category_id"] for rec in pred] == [9]


def test_category_filter_keeps_negative_images(tmp_path):
    """Images without the selected object stay in GT so FPs still count."""

    mask5 = _mask(y0=4, x0=5)
    mask9 = _mask(y0=15, x0=20)
    scene = _write_bop_scene(tmp_path, mask5, obj_id=5)
    Image.fromarray((mask9.astype(np.uint8) * 255)).save(
        scene / "mask_visib" / "000008_000000.png")
    Image.fromarray(np.zeros((*mask9.shape, 3), dtype=np.uint8)).save(
        scene / "rgb" / "000008.png")
    (scene / "scene_gt.json").write_text(json.dumps({
        "7": [{"obj_id": 5}],
        "8": [{"obj_id": 9}],
    }))
    det_path = tmp_path / "detections.json"
    # Hallucinated cat-9 proposal on the negative frame (im 7 has only obj 5).
    det_path.write_text(json.dumps([
        {
            "scene_id": 1,
            "image_id": 7,
            "category_id": 9,
            "score": 0.95,
            "segmentation": _rle(mask5),
        },
        {
            "scene_id": 1,
            "image_id": 8,
            "category_id": 9,
            "score": 0.9,
            "segmentation": _rle(mask9),
        },
    ]))

    coco_gt = build_coco_gt_from_bop(tmp_path, category_ids={9})
    pred = detections_to_coco_results(det_path, coco_gt, category_ids={9})

    image_ids = {im["id"] for im in coco_gt["images"]}
    assert image_ids == {bop_image_id(1, 7), bop_image_id(1, 8)}
    assert [ann["category_id"] for ann in coco_gt["annotations"]] == [9]
    assert sorted(rec["image_id"] for rec in pred) == [
        bop_image_id(1, 7), bop_image_id(1, 8),
    ]


def test_missing_visible_mask_is_loud(tmp_path):
    scene = tmp_path / "test" / "000001"
    scene.mkdir(parents=True)
    (scene / "scene_gt.json").write_text(json.dumps({
        "7": [{"obj_id": 5}]
    }))

    with pytest.raises(FileNotFoundError, match="missing BOP GT mask"):
        build_coco_gt_from_bop(tmp_path)


def test_mask_kind_does_not_silently_fall_back(tmp_path):
    """Only the requested mask directory is used — no mask_visib ↔ mask swap."""

    gt_mask = _mask()
    scene = tmp_path / "test" / "000001"
    (scene / "mask").mkdir(parents=True)
    (scene / "rgb").mkdir()
    Image.fromarray(np.zeros((*gt_mask.shape, 3), dtype=np.uint8)).save(
        scene / "rgb" / "000007.png")
    Image.fromarray((gt_mask.astype(np.uint8) * 255)).save(
        scene / "mask" / "000007_000000.png")
    (scene / "scene_gt.json").write_text(json.dumps({
        "7": [{"obj_id": 5}]
    }))

    with pytest.raises(FileNotFoundError, match="mask_kind=mask_visib"):
        build_coco_gt_from_bop(tmp_path, mask_kind="mask_visib")

    coco_gt = build_coco_gt_from_bop(tmp_path, mask_kind="mask")
    assert len(coco_gt["annotations"]) == 1


def test_filter_coco_gt_by_targets_and_categories(tmp_path):
    mask5 = _mask(y0=4, x0=5)
    mask9 = _mask(y0=15, x0=20)
    scene = _write_bop_scene(tmp_path, mask5, obj_id=5)
    Image.fromarray((mask9.astype(np.uint8) * 255)).save(
        scene / "mask_visib" / "000007_000001.png")
    (scene / "scene_gt.json").write_text(json.dumps({
        "7": [{"obj_id": 5}, {"obj_id": 9}]
    }))

    full_gt = build_coco_gt_from_bop(tmp_path)
    assert len(full_gt["annotations"]) == 2

    filtered = filter_coco_gt(
        full_gt,
        targets={(1, 7, 5)},
        category_ids={5, 9},
    )
    assert [ann["category_id"] for ann in filtered["annotations"]] == [5]
    assert [im["id"] for im in filtered["images"]] == [bop_image_id(1, 7)]
    assert [cat["id"] for cat in filtered["categories"]] == [5]


def test_filter_coco_gt_category_only_keeps_negative_images(tmp_path):
    mask5 = _mask(y0=4, x0=5)
    mask9 = _mask(y0=15, x0=20)
    scene = _write_bop_scene(tmp_path, mask5, obj_id=5)
    Image.fromarray((mask9.astype(np.uint8) * 255)).save(
        scene / "mask_visib" / "000008_000000.png")
    Image.fromarray(np.zeros((*mask9.shape, 3), dtype=np.uint8)).save(
        scene / "rgb" / "000008.png")
    (scene / "scene_gt.json").write_text(json.dumps({
        "7": [{"obj_id": 5}],
        "8": [{"obj_id": 9}],
    }))

    full_gt = build_coco_gt_from_bop(tmp_path)
    filtered = filter_coco_gt(full_gt, category_ids={9})
    assert [ann["category_id"] for ann in filtered["annotations"]] == [9]
    assert {im["id"] for im in filtered["images"]} == {
        bop_image_id(1, 7), bop_image_id(1, 8),
    }


def test_filter_coco_gt_targets_requires_bop_fields():
    bare = {
        "info": {},
        "licenses": [],
        "images": [{"id": 1, "width": 10, "height": 10}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 5,
            "segmentation": {"size": [10, 10], "counts": "1"},
            "area": 1, "bbox": [0, 0, 1, 1], "iscrowd": 0,
        }],
        "categories": [{"id": 5, "name": "obj_000005"}],
    }
    with pytest.raises(ValueError, match="scene_id and bop_image_id"):
        filter_coco_gt(bare, targets={(1, 7, 5)})


def test_bop_seg_eval_cli_writes_summary(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples" / "bop_seg_eval.py"
    spec = importlib.util.spec_from_file_location("bop_seg_eval", example)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    gt_mask = _mask()
    _write_bop_scene(tmp_path, gt_mask)
    det_path = tmp_path / "detections.json"
    targets_path = tmp_path / "targets.json"
    out_dir = tmp_path / "seg_eval"
    _write_detections(det_path, gt_mask)
    targets_path.write_text(json.dumps([
        {"scene_id": 1, "im_id": 7, "obj_id": 5, "inst_count": 1},
    ]))

    rc = mod.main([
        "--bop", str(tmp_path),
        "--detections", str(det_path),
        "--targets", str(targets_path),
        "--out-dir", str(out_dir),
    ])

    summary = json.loads((out_dir / "summary.json").read_text())
    assert rc == 0
    assert summary["gt_images"] == 1
    assert summary["gt_annotations"] == 1
    assert summary["predictions"] == 1
    assert summary["stats"]["AP"] == pytest.approx(1.0)


def test_bop_seg_eval_cli_gt_coco_filters_targets(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples" / "bop_seg_eval.py"
    spec = importlib.util.spec_from_file_location("bop_seg_eval", example)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mask5 = _mask(y0=4, x0=5)
    mask9 = _mask(y0=15, x0=20)
    scene = _write_bop_scene(tmp_path, mask5, obj_id=5)
    Image.fromarray((mask9.astype(np.uint8) * 255)).save(
        scene / "mask_visib" / "000007_000001.png")
    (scene / "scene_gt.json").write_text(json.dumps({
        "7": [{"obj_id": 5}, {"obj_id": 9}]
    }))

    full_gt = build_coco_gt_from_bop(tmp_path)
    gt_path = tmp_path / "prebuilt_gt.json"
    write_json(gt_path, full_gt)

    det_path = tmp_path / "detections.json"
    det_path.write_text(json.dumps([
        {
            "scene_id": 1,
            "image_id": 7,
            "category_id": 5,
            "score": 0.9,
            "segmentation": _rle(mask5),
        },
        {
            "scene_id": 1,
            "image_id": 7,
            "category_id": 9,
            "score": 0.8,
            "segmentation": _rle(mask9),
        },
    ]))
    targets_path = tmp_path / "targets.json"
    targets_path.write_text(json.dumps([
        {"scene_id": 1, "im_id": 7, "obj_id": 5, "inst_count": 1},
    ]))
    out_dir = tmp_path / "seg_eval_gt_coco"

    rc = mod.main([
        "--gt-coco", str(gt_path),
        "--detections", str(det_path),
        "--targets", str(targets_path),
        "--out-dir", str(out_dir),
    ])

    written_gt = json.loads((out_dir / "gt_coco.json").read_text())
    summary = json.loads((out_dir / "summary.json").read_text())
    assert rc == 0
    assert [ann["category_id"] for ann in written_gt["annotations"]] == [5]
    assert summary["gt_annotations"] == 1
    assert summary["predictions"] == 1
    assert summary["stats"]["AP"] == pytest.approx(1.0)
