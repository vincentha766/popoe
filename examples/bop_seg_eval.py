"""Evaluate BOP-format segmentation detections with COCO mask AP.

This is the mask-proposal counterpart to ``examples/bop_eval.py``: it never
runs pose estimation, it only scores the 2D ``segmentation`` field in a
detections JSON. The ground truth is built from BOP visible masks
(``mask_visib``) and evaluated with ``pycocotools.COCOeval(..., "segm")``.

Example:
  python examples/bop_seg_eval.py \
      --bop /path/to/ycbv \
      --detections data/detections/cnos/cnos-fastsam_ycbv-test.json \
      --targets /path/to/ycbv/test_targets_bop19.json \
      --out-dir outputs/ycbv_cnos_seg_ap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from popoe.segmentation_eval import (
    DEFAULT_IMAGE_ID_FACTOR,
    build_coco_gt_from_bop,
    detections_to_coco_results,
    evaluate_coco_segm,
    filter_coco_gt,
    load_bop_targets,
    write_json,
)


def _parse_obj_ids(text: str) -> set[int]:
    return {int(x) for x in text.split(",") if x.strip()}


def _load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Score BOP detections JSON masks with COCO segm AP.")
    ap.add_argument("--bop", default="",
                    help="BOP dataset root, e.g. /data/ycbv. Required unless "
                         "--gt-coco is a merged GT JSON with matching image ids.")
    ap.add_argument("--split", default="test",
                    help="BOP split to read when building GT from --bop.")
    ap.add_argument("--gt-coco", default="",
                    help="optional prebuilt merged COCO GT JSON. Its image ids "
                         "must match scene_id * --image-id-factor + im_id.")
    ap.add_argument("--detections", required=True,
                    help="BOP-format detections JSON with segmentation masks.")
    ap.add_argument("--targets", default="",
                    help="optional BOP targets JSON; filters GT and detections "
                         "to target (scene_id, im_id, obj_id) triples. With "
                         "--gt-coco, images must carry scene_id and bop_image_id.")
    ap.add_argument("--objs", default="",
                    help="optional comma-separated obj ids, e.g. 5,8,9; "
                         "filters GT categories and detections.")
    ap.add_argument("--out-dir", default="outputs/bop_seg_eval",
                    help="directory for gt_coco.json, pred_coco.json, summary.json.")
    ap.add_argument("--mask-kind", default="mask_visib",
                    choices=["mask_visib", "mask"],
                    help="BOP GT mask directory to use when building GT.")
    ap.add_argument("--image-id-factor", type=int,
                    default=DEFAULT_IMAGE_ID_FACTOR,
                    help="COCO image id = scene_id * factor + im_id.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_ids = _parse_obj_ids(args.objs)
    targets = load_bop_targets(args.targets or None)
    if obj_ids and targets is not None:
        targets = {t for t in targets if t[2] in obj_ids}

    if args.gt_coco:
        coco_gt = _load_json(args.gt_coco)
        # Match build_coco_gt_from_bop filtering so --targets / --objs apply
        # to prebuilt GT, not only to detections.
        if targets is not None or obj_ids:
            try:
                coco_gt = filter_coco_gt(
                    coco_gt,
                    targets=targets,
                    category_ids=obj_ids or None,
                )
            except ValueError as e:
                raise SystemExit(str(e))
    else:
        if not args.bop:
            raise SystemExit("pass --bop, or pass --gt-coco with matching image ids")
        try:
            coco_gt = build_coco_gt_from_bop(
                args.bop,
                split=args.split,
                targets=targets,
                category_ids=obj_ids or None,
                image_id_factor=args.image_id_factor,
                mask_kind=args.mask_kind,
            )
        except FileNotFoundError as e:
            raise SystemExit(str(e))

    category_ids = sorted(obj_ids) if obj_ids else None
    image_ids = sorted(int(im["id"]) for im in coco_gt.get("images", []))
    if not image_ids:
        raise SystemExit("ground truth contains no images after filtering")
    if not coco_gt.get("annotations"):
        raise SystemExit("ground truth contains no annotations after filtering")

    pred = detections_to_coco_results(
        args.detections,
        coco_gt,
        image_id_factor=args.image_id_factor,
        targets=targets,
        category_ids=obj_ids or None,
    )

    gt_path = out_dir / "gt_coco.json"
    pred_path = out_dir / "pred_coco.json"
    write_json(gt_path, coco_gt)
    write_json(pred_path, pred)

    summary = {
        "detections": args.detections,
        "bop": args.bop or None,
        "split": args.split,
        "targets": args.targets or None,
        "objs": sorted(obj_ids),
        "gt_images": len(coco_gt.get("images", [])),
        "gt_annotations": len(coco_gt.get("annotations", [])),
        "predictions": len(pred),
        "gt_coco": str(gt_path),
        "pred_coco": str(pred_path),
    }
    if not pred:
        summary["stats"] = None
        write_json(out_dir / "summary.json", summary)
        raise SystemExit(
            "no predictions matched the GT image/category set; wrote "
            f"{pred_path} for inspection"
        )

    stats = evaluate_coco_segm(
        gt_path,
        pred_path,
        image_ids=image_ids,
        category_ids=category_ids,
    )
    summary["stats"] = stats
    write_json(out_dir / "summary.json", summary)

    print(f"wrote {gt_path}")
    print(f"wrote {pred_path}")
    print(f"wrote {out_dir / 'summary.json'}")
    print("segm AP: "
          f"AP={stats['AP']:.4f} AP50={stats['AP50']:.4f} "
          f"AP75={stats['AP75']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
