"""COCO-style mask AP evaluation for BOP-format segmentation detections.

The BOP model-based 2D segmentation task evaluates records shaped like
``{scene_id, image_id, category_id, score, segmentation}`` against visible
object masks. popoe already loads those detections for pose estimation; this
module adds the thin COCOeval bridge needed to score the masks themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from popoe.segmentor_detections import (
    decode_detection_mask,
    load_bop_detections,
)


DEFAULT_IMAGE_ID_FACTOR = 1_000_000


def bop_image_id(scene_id: int, im_id: int,
                 factor: int = DEFAULT_IMAGE_ID_FACTOR) -> int:
    """Map a BOP ``(scene_id, image_id)`` pair to one COCO image id."""

    return int(scene_id) * int(factor) + int(im_id)


def _read_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def _bbox_from_mask(mask: np.ndarray) -> list[float]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return [0.0, 0.0, 0.0, 0.0]
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def _encode_coco_rle(mask: np.ndarray) -> dict:
    from pycocotools import mask as coco_mask

    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def _read_mask(path: Path) -> np.ndarray:
    from PIL import Image

    return (np.asarray(Image.open(path)) > 0).astype(bool)


def _mask_path(scene_dir: Path, im_id: int, gt_idx: int,
               kind: str = "mask_visib") -> Path:
    name = f"{int(im_id):06d}_{int(gt_idx):06d}.png"
    preferred = scene_dir / kind / name
    if preferred.exists():
        return preferred
    fallback_kind = "mask" if kind == "mask_visib" else "mask_visib"
    fallback = scene_dir / fallback_kind / name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"missing BOP GT mask for scene={scene_dir.name} im={im_id} "
        f"gt_idx={gt_idx}: tried {preferred} and {fallback}"
    )


def load_bop_targets(path: str | Path | None) -> set[tuple[int, int, int]] | None:
    """Load BOP target triples ``(scene_id, im_id, obj_id)``.

    The target file may contain repeated rows when ``inst_count`` > 1. For mask
    AP filtering we only need the image/object triple.
    """

    if path is None:
        return None
    out = set()
    for rec in _read_json(path):
        out.add((int(rec["scene_id"]), int(rec["im_id"]), int(rec["obj_id"])))
    return out


def _scene_dirs(bop_root: Path, split: str) -> list[Path]:
    split_dir = bop_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"BOP split directory not found: {split_dir}")
    return sorted(p for p in split_dir.iterdir()
                  if p.is_dir() and p.name.isdigit())


def build_coco_gt_from_bop(
    bop_root: str | Path,
    *,
    split: str = "test",
    targets: set[tuple[int, int, int]] | None = None,
    category_ids: set[int] | None = None,
    image_id_factor: int = DEFAULT_IMAGE_ID_FACTOR,
    mask_kind: str = "mask_visib",
) -> dict:
    """Build a COCO instance-segmentation GT dict from a BOP dataset split.

    Requires BOP visible-mask files such as
    ``test/000048/mask_visib/000001_000000.png``. If a local dataset only has
    sparse RGB + poses, this function fails loudly rather than reporting fake
    AP.
    """

    bop_root = Path(bop_root)
    category_filter = {int(cid) for cid in category_ids} if category_ids else None
    images: dict[int, dict] = {}
    annotations: list[dict] = []
    seen_category_ids: set[int] = set()
    ann_id = 1

    for scene_dir in _scene_dirs(bop_root, split):
        scene_id = int(scene_dir.name)
        gt_path = scene_dir / "scene_gt.json"
        if not gt_path.exists():
            continue
        scene_gt = _read_json(gt_path)
        for im_key, objs in scene_gt.items():
            im_id = int(im_key)
            for gt_idx, obj in enumerate(objs):
                obj_id = int(obj["obj_id"])
                if targets is not None and (scene_id, im_id, obj_id) not in targets:
                    continue
                if category_filter is not None and obj_id not in category_filter:
                    continue
                mask = _read_mask(_mask_path(scene_dir, im_id, gt_idx, mask_kind))
                if not mask.any():
                    continue
                image_id = bop_image_id(scene_id, im_id, image_id_factor)
                if image_id not in images:
                    images[image_id] = {
                        "id": image_id,
                        "scene_id": scene_id,
                        "bop_image_id": im_id,
                        "file_name": f"{split}/{scene_id:06d}/rgb/{im_id:06d}.png",
                        "width": int(mask.shape[1]),
                        "height": int(mask.shape[0]),
                    }
                rle = _encode_coco_rle(mask)
                bbox = _bbox_from_mask(mask)
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": obj_id,
                    "segmentation": rle,
                    "area": int(mask.sum()),
                    "bbox": bbox,
                    "iscrowd": 0,
                })
                ann_id += 1
                seen_category_ids.add(obj_id)

    categories = [
        {"id": int(cid), "name": f"obj_{int(cid):06d}"}
        for cid in sorted(seen_category_ids)
    ]
    return {
        "info": {"description": f"BOP {bop_root.name} {split} visible masks"},
        "licenses": [],
        "images": [images[k] for k in sorted(images)],
        "annotations": annotations,
        "categories": categories,
    }


def coco_image_keyset(coco_gt: Mapping) -> set[tuple[int, int]]:
    """Return BOP ``(scene_id, image_id)`` pairs available in a COCO GT dict."""

    pairs = set()
    for im in coco_gt.get("images", []):
        if "scene_id" in im and "bop_image_id" in im:
            pairs.add((int(im["scene_id"]), int(im["bop_image_id"])))
    return pairs


def filter_coco_gt(
    coco_gt: Mapping,
    *,
    targets: set[tuple[int, int, int]] | None = None,
    category_ids: set[int] | None = None,
) -> dict:
    """Filter a prebuilt COCO GT dict by BOP targets and/or category ids.

    Matches the filtering applied by ``build_coco_gt_from_bop`` so ``--gt-coco``
    + ``--targets`` / ``--objs`` behave the same as building GT from ``--bop``.

    When ``targets`` is set, every image must carry ``scene_id`` and
    ``bop_image_id`` (as written by ``build_coco_gt_from_bop``).
    """

    if targets is None and not category_ids:
        return {
            "info": dict(coco_gt.get("info", {})),
            "licenses": list(coco_gt.get("licenses", [])),
            "images": list(coco_gt.get("images", [])),
            "annotations": list(coco_gt.get("annotations", [])),
            "categories": list(coco_gt.get("categories", [])),
        }

    category_filter = {int(cid) for cid in category_ids} if category_ids else None
    image_meta: dict[int, tuple[int, int]] | None = None
    if targets is not None:
        image_meta = {}
        for im in coco_gt.get("images", []):
            if "scene_id" not in im or "bop_image_id" not in im:
                raise ValueError(
                    "filtering prebuilt COCO GT by targets requires each image "
                    "to have scene_id and bop_image_id fields (as written by "
                    "build_coco_gt_from_bop)"
                )
            image_meta[int(im["id"])] = (
                int(im["scene_id"]), int(im["bop_image_id"]))

    annotations: list[dict] = []
    for ann in coco_gt.get("annotations", []):
        obj_id = int(ann["category_id"])
        image_id = int(ann["image_id"])
        if category_filter is not None and obj_id not in category_filter:
            continue
        if image_meta is not None:
            if image_id not in image_meta:
                continue
            scene_id, im_id = image_meta[image_id]
            if (scene_id, im_id, obj_id) not in targets:
                continue
        annotations.append(dict(ann))

    kept_image_ids = {int(a["image_id"]) for a in annotations}
    kept_cat_ids = {int(a["category_id"]) for a in annotations}
    images = [dict(im) for im in coco_gt.get("images", [])
              if int(im["id"]) in kept_image_ids]
    categories = [dict(c) for c in coco_gt.get("categories", [])
                  if int(c["id"]) in kept_cat_ids]
    have = {int(c["id"]) for c in categories}
    for cid in sorted(kept_cat_ids - have):
        categories.append({"id": int(cid), "name": f"obj_{int(cid):06d}"})
    categories.sort(key=lambda c: int(c["id"]))
    return {
        "info": dict(coco_gt.get("info", {})),
        "licenses": list(coco_gt.get("licenses", [])),
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def detections_to_coco_results(
    detections_json: str | Path,
    coco_gt: Mapping,
    *,
    image_id_factor: int = DEFAULT_IMAGE_ID_FACTOR,
    targets: set[tuple[int, int, int]] | None = None,
    category_ids: set[int] | None = None,
) -> list[dict]:
    """Convert BOP-format detections to COCO results aligned to ``coco_gt``."""

    gt_image_ids = {int(im["id"]) for im in coco_gt.get("images", [])}
    gt_cat_ids = {int(c["id"]) for c in coco_gt.get("categories", [])}
    category_filter = {int(cid) for cid in category_ids} if category_ids else None
    out = []
    for rec in load_bop_detections(str(detections_json)):
        scene_id = int(rec["scene_id"])
        im_id = int(rec["image_id"])
        obj_id = int(rec["category_id"])
        if targets is not None and (scene_id, im_id, obj_id) not in targets:
            continue
        if category_filter is not None and obj_id not in category_filter:
            continue
        image_id = bop_image_id(scene_id, im_id, image_id_factor)
        if image_id not in gt_image_ids or obj_id not in gt_cat_ids:
            continue
        mask = decode_detection_mask(rec["segmentation"])
        if not mask.any():
            continue
        bbox = rec.get("bbox")
        if bbox is None:
            bbox = _bbox_from_mask(mask)
        out.append({
            "image_id": image_id,
            "category_id": obj_id,
            "score": float(rec["score"]),
            "segmentation": _encode_coco_rle(mask),
            "bbox": [float(x) for x in bbox],
        })
    return out


COCO_STATS = [
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def evaluate_coco_segm(
    coco_gt_json: str | Path,
    coco_results_json: str | Path,
    *,
    image_ids: Sequence[int] | None = None,
    category_ids: Sequence[int] | None = None,
) -> dict:
    """Run COCOeval in ``segm`` mode and return named summary stats."""

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(coco_gt_json))
    coco_dt = coco_gt.loadRes(str(coco_results_json))
    ev = COCOeval(coco_gt, coco_dt, "segm")
    if image_ids is not None:
        ev.params.imgIds = [int(x) for x in image_ids]
    if category_ids is not None:
        ev.params.catIds = [int(x) for x in category_ids]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {name: float(value) for name, value in zip(COCO_STATS, ev.stats)}


def write_json(path: str | Path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)
