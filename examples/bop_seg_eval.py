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
import csv
import json
from pathlib import Path

from popoe.segmentation_eval import (
    DEFAULT_IMAGE_ID_FACTOR,
    build_coco_gt_from_bop,
    detections_to_coco_results,
    evaluate_coco_segm,
    evaluate_coco_segm_per_category,
    filter_coco_gt,
    load_bop_model_category_ids,
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
                    help="optional BOP targets JSON; target images select GT "
                         "and detections, matching the official BOP COCO "
                         "evaluator. With --gt-coco, images must carry scene_id "
                         "and bop_image_id.")
    ap.add_argument("--objs", default="",
                    help="optional comma-separated obj ids, e.g. 5,8,9; "
                         "filters GT categories and detections. With "
                         "--targets, this does not shrink the target image set.")
    ap.add_argument("--out-dir", default="outputs/bop_seg_eval",
                    help="directory for gt_coco.json, pred_coco.json, summary.json.")
    ap.add_argument("--mask-kind", default="mask_visib",
                    choices=["mask_visib", "mask"],
                    help="BOP GT mask directory to use when building GT.")
    ap.add_argument("--category-source", default="observed",
                    choices=["observed", "models_info"],
                    help="COCO categories source when building GT from --bop. "
                         "models_info reads models/models_info.json for "
                         "stricter BOP parity.")
    ap.add_argument("--image-id-factor", type=int,
                    default=DEFAULT_IMAGE_ID_FACTOR,
                    help="COCO image id = scene_id * factor + im_id.")
    ap.add_argument("--per-object", action="store_true",
                    help="also write per_object.json and per_object.csv with "
                         "COCO segm stats for each category in this source.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_ids = _parse_obj_ids(args.objs)
    targets = load_bop_targets(args.targets or None)

    if args.gt_coco:
        if args.category_source != "observed":
            raise SystemExit("--category-source requires building GT from --bop")
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
            category_source_ids = (
                load_bop_model_category_ids(args.bop)
                if args.category_source == "models_info" else None
            )
        except FileNotFoundError as e:
            raise SystemExit(str(e))
        try:
            coco_gt = build_coco_gt_from_bop(
                args.bop,
                split=args.split,
                targets=targets,
                category_ids=obj_ids or None,
                category_source_ids=category_source_ids,
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
        "category_source": args.category_source,
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
    if args.per_object:
        per_object = evaluate_coco_segm_per_category(
            gt_path,
            pred_path,
            image_ids=image_ids,
            category_ids=category_ids,
        )
        per_object_json = out_dir / "per_object.json"
        per_object_csv = out_dir / "per_object.csv"
        write_json(per_object_json, per_object)
        with open(per_object_csv, "w", newline="") as f:
            fieldnames = [
                "category_id", "name", "gt_annotations", "gt_images",
                "predictions", "pred_images",
                "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
            ]
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            wr.writerows(per_object)
        summary["per_object_json"] = str(per_object_json)
        summary["per_object_csv"] = str(per_object_csv)
    write_json(out_dir / "summary.json", summary)

    print(f"wrote {gt_path}")
    print(f"wrote {pred_path}")
    print(f"wrote {out_dir / 'summary.json'}")
    if args.per_object:
        print(f"wrote {out_dir / 'per_object.json'}")
        print(f"wrote {out_dir / 'per_object.csv'}")
    print("segm AP: "
          f"AP={stats['AP']:.4f} AP50={stats['AP50']:.4f} "
          f"AP75={stats['AP75']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
