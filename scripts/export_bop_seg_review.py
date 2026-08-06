#!/usr/bin/env python3
"""Export BOP segmentation review panels for human verification (CPU-only).

For each selected (source, scene, image, category) draws:
  - RGB + mask overlay
  - binary mask
  - tight crop around the mask
  - optional GT mask_visib overlay (IoU printed when GT is present)

Sampling (default):
  * --per-source N random (scene, image) frames that have detections
  * --worst-per-obj K lowest top-1 IoU frames per object (needs mask_visib)
  * --force-scenes  always include these scene ids (default: 48 for YCB-V clamps)

Does not run any segmentor — only consumes existing detections JSON files.
source= names are stamped into the output tree for provenance (cnos / muse / …).

Example:
  uv run python scripts/export_bop_seg_review.py \\
    --bop bop_data/ycbv \\
    --source cnos=data/detections/cnos/cnos-fastsam_ycbv-test.json \\
    --source muse=outputs/seg_ap_…/official_submissions/muse-full_ycbv-test_official.json \\
    --out-dir outputs/pipeline_verify_seg_vis/ycbv \\
    --per-source 8 --worst-per-obj 1 --topk 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from popoe.segmentor_detections import decode_detection_mask, load_bop_detections

COLORS = [
    (255, 64, 64),
    (64, 220, 80),
    (64, 128, 255),
    (255, 192, 64),
    (220, 64, 255),
    (64, 220, 220),
    (255, 128, 32),
    (180, 180, 40),
]


def _parse_source(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f"--source must be name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    name, path = name.strip(), path.strip()
    if not name or not path:
        raise SystemExit(f"bad --source {spec!r}")
    if not os.path.isfile(path):
        raise SystemExit(f"detections file not found: {path}")
    return name, path


def _rgb_path(bop: Path, scene_id: int, im_id: int) -> Path:
    return bop / "test" / f"{scene_id:06d}" / "rgb" / f"{im_id:06d}.png"


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _gt_masks_for_image(bop: Path, scene_id: int, im_id: int, category_id: int):
    """Return list of bool GT mask_visib arrays for this (scene, im, obj)."""
    scene_dir = bop / "test" / f"{scene_id:06d}"
    gt_path = scene_dir / "scene_gt.json"
    if not gt_path.is_file():
        return []
    gts = json.loads(gt_path.read_text())
    rows = gts.get(str(im_id), gts.get(im_id, []))
    out = []
    for inst_idx, row in enumerate(rows):
        if int(row["obj_id"]) != int(category_id):
            continue
        # BOP mask_visib naming: {im_id:06d}_{inst_idx:06d}.png
        mp = scene_dir / "mask_visib" / f"{im_id:06d}_{inst_idx:06d}.png"
        if not mp.is_file():
            continue
        arr = np.asarray(Image.open(mp))
        if arr.ndim == 3:
            arr = arr[..., 0]
        out.append(arr > 0)
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _best_gt_iou(pred: np.ndarray, gts: list[np.ndarray]) -> float:
    if not gts:
        return float("nan")
    return max(_iou(pred, g) for g in gts)


def _overlay(rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45) -> np.ndarray:
    out = rgb.copy()
    if mask.shape[:2] != rgb.shape[:2]:
        return out
    c = np.array(color, dtype=np.float32)
    m = mask.astype(bool)
    out[m] = (out[m].astype(np.float32) * (1 - alpha) + c * alpha).astype(np.uint8)
    # contour
    try:
        import cv2

        # CHAIN_APPROX_NONE, width 1: SIMPLE discards collinear points, so
        # concavities get straightened away from the real boundary, and a
        # width-2 stroke straddles the edge with half of it outside the mask.
        # Together they make a pixel-accurate mask look like it misses the object.
        cnts, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(out, cnts, -1, tuple(int(x) for x in color), 1)
    except Exception:
        pass
    return out


def _crop(rgb: np.ndarray, mask: np.ndarray, pad: int = 12) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return rgb
    y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad + 1)
    return rgb[y0:y1, x0:x1]


def _annotate(img: np.ndarray, lines: list[str]) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    y = 4
    for line in lines:
        draw.rectangle([2, y, 2 + 7 * len(line) + 8, y + 14], fill=(0, 0, 0))
        draw.text((6, y), line, fill=(255, 255, 255), font=font)
        y += 16
    return np.asarray(pil)


def _pick_keys(
    by_image: dict[tuple[int, int], list[dict]],
    per_source: int,
    force_scenes: set[int],
    rng: random.Random,
) -> list[tuple[int, int]]:
    keys = list(by_image.keys())
    forced = [k for k in keys if k[0] in force_scenes]
    rest = [k for k in keys if k[0] not in force_scenes]
    rng.shuffle(rest)
    # one random image per forced scene if multiple
    forced_pick = []
    seen_sc = set()
    for sc, im in forced:
        if sc in seen_sc:
            continue
        forced_pick.append((sc, im))
        seen_sc.add(sc)
    n_rand = max(0, per_source)
    picked = forced_pick + rest[:n_rand]
    # de-dupe preserve order
    out, seen = [], set()
    for k in picked:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def export_source(
    *,
    name: str,
    path: str,
    bop: Path,
    out_root: Path,
    per_source: int,
    topk: int,
    force_scenes: set[int],
    worst_per_obj: int,
    objs: set[int] | None,
    rng: random.Random,
) -> list[dict]:
    print(f"[{name}] loading {path} …", flush=True)
    recs = load_bop_detections(path, source=name)
    if objs:
        recs = [r for r in recs if int(r["category_id"]) in objs]

    by_image: dict[tuple[int, int], list[dict]] = defaultdict(list)
    by_obj: dict[int, list[dict]] = defaultdict(list)
    for r in recs:
        key = (int(r["scene_id"]), int(r["image_id"]))
        by_image[key].append(r)
        by_obj[int(r["category_id"])].append(r)

    for key in by_image:
        by_image[key].sort(key=lambda d: -float(d["score"]))

    keys = _pick_keys(by_image, per_source, force_scenes, rng)

    # Pre-score top-1 IoU for worst-per-obj (only on images with RGB + GT)
    worst_extra: list[tuple[int, int]] = []
    if worst_per_obj > 0:
        print(f"[{name}] scoring top-1 IoU for worst-per-obj …", flush=True)
        for obj_id, rows in by_obj.items():
            scored = []
            # group by image, take top-1
            per_im: dict[tuple[int, int], dict] = {}
            for r in rows:
                k = (int(r["scene_id"]), int(r["image_id"]))
                if k not in per_im or float(r["score"]) > float(per_im[k]["score"]):
                    per_im[k] = r
            for k, r in per_im.items():
                rgb_p = _rgb_path(bop, k[0], k[1])
                if not rgb_p.is_file():
                    continue
                try:
                    mask = decode_detection_mask(r["segmentation"])
                except Exception:
                    continue
                gts = _gt_masks_for_image(bop, k[0], k[1], obj_id)
                iou = _best_gt_iou(mask, gts)
                if np.isnan(iou):
                    continue
                scored.append((iou, k))
            scored.sort(key=lambda t: t[0])
            for _, k in scored[:worst_per_obj]:
                worst_extra.append(k)

    for k in worst_extra:
        if k not in keys:
            keys.append(k)

    index_rows: list[dict] = []
    src_dir = out_root / name
    src_dir.mkdir(parents=True, exist_ok=True)

    for scene_id, im_id in keys:
        rgb_p = _rgb_path(bop, scene_id, im_id)
        if not rgb_p.is_file():
            print(f"  skip missing RGB {rgb_p}", flush=True)
            continue
        rgb = _load_rgb(rgb_p)
        rows = by_image[(scene_id, im_id)]
        # topk per category
        per_cat: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            cid = int(r["category_id"])
            if objs and cid not in objs:
                continue
            if len(per_cat[cid]) < topk:
                per_cat[cid].append(r)

        for cid, cat_rows in sorted(per_cat.items()):
            gts = _gt_masks_for_image(bop, scene_id, im_id, cid)
            for rank, r in enumerate(cat_rows, start=1):
                try:
                    mask = decode_detection_mask(r["segmentation"])
                except Exception as e:
                    print(f"  decode fail s{scene_id} i{im_id} c{cid}: {e}", flush=True)
                    continue
                if mask.shape[:2] != rgb.shape[:2]:
                    print(
                        f"  shape mismatch s{scene_id} i{im_id} c{cid}: "
                        f"mask {mask.shape} rgb {rgb.shape}",
                        flush=True,
                    )
                    continue
                iou = _best_gt_iou(mask, gts)
                color = COLORS[(cid + rank) % len(COLORS)]
                ov = _overlay(rgb, mask, color)
                # faint GT contour in white when available
                if gts:
                    try:
                        import cv2

                        for g in gts:
                            if g.shape[:2] != rgb.shape[:2]:
                                continue
                            cnts, _ = cv2.findContours(
                                g.astype(np.uint8),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE,
                            )
                            cv2.drawContours(ov, cnts, -1, (255, 255, 255), 1)
                    except Exception:
                        pass
                label = [
                    f"{name} obj={cid} rank={rank}",
                    f"s={scene_id} im={im_id} score={float(r['score']):.3f}",
                ]
                if not np.isnan(iou):
                    label.append(f"IoU(gt)={iou:.3f}")
                ov = _annotate(ov, label)
                crop = _crop(_overlay(rgb, mask, color), mask)
                stem = f"s{scene_id:06d}_im{im_id:06d}_obj{cid:02d}_r{rank}"
                base = src_dir / stem
                Image.fromarray(ov).save(f"{base}_overlay.png")
                Image.fromarray((mask.astype(np.uint8) * 255)).save(f"{base}_mask.png")
                Image.fromarray(crop).save(f"{base}_crop.png")
                index_rows.append(
                    {
                        "source": name,
                        "scene_id": scene_id,
                        "image_id": im_id,
                        "category_id": cid,
                        "rank": rank,
                        "score": float(r["score"]),
                        "iou_gt": None if np.isnan(iou) else round(iou, 4),
                        "overlay": str(Path(name) / f"{stem}_overlay.png"),
                        "mask": str(Path(name) / f"{stem}_mask.png"),
                        "crop": str(Path(name) / f"{stem}_crop.png"),
                    }
                )
    return index_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bop", required=True, help="BOP dataset root (…/ycbv or …/lmo)")
    ap.add_argument(
        "--source",
        action="append",
        default=[],
        help="name=path to detections JSON; repeatable",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-source", type=int, default=10, help="random frames per source")
    ap.add_argument("--topk", type=int, default=1, help="top-K detections per category per image")
    ap.add_argument(
        "--worst-per-obj",
        type=int,
        default=2,
        help="also export K lowest top-1 IoU frames per object (0 to disable)",
    )
    ap.add_argument(
        "--force-scenes",
        default="",
        help="comma-separated scene ids always included (default: 48 if present)",
    )
    ap.add_argument("--objs", default="", help="optional comma-separated object ids filter")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.source:
        raise SystemExit("provide at least one --source name=path")

    bop = Path(args.bop)
    if not (bop / "test").is_dir():
        raise SystemExit(f"--bop must contain test/: {bop}")

    if args.force_scenes.strip():
        force = {int(x) for x in args.force_scenes.split(",") if x.strip()}
    else:
        # default clamp scene on YCB-V; harmless if absent
        force = {48}

    objs = None
    if args.objs.strip():
        objs = {int(x) for x in args.objs.split(",") if x.strip()}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    all_rows: list[dict] = []
    for spec in args.source:
        name, path = _parse_source(spec)
        rows = export_source(
            name=name,
            path=path,
            bop=bop,
            out_root=out,
            per_source=args.per_source,
            topk=args.topk,
            force_scenes=force,
            worst_per_obj=args.worst_per_obj,
            objs=objs,
            rng=rng,
        )
        all_rows.extend(rows)
        print(f"[{name}] wrote {len(rows)} panels", flush=True)

    # INDEX
    index_json = out / "index.json"
    index_json.write_text(json.dumps(all_rows, indent=2))
    md = ["# Segmentation review index", "", f"BOP root: `{bop}`", "", "| source | scene | im | obj | rank | score | IoU(gt) | overlay |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in all_rows:
        iou = "" if r["iou_gt"] is None else f"{r['iou_gt']:.3f}"
        md.append(
            f"| {r['source']} | {r['scene_id']} | {r['image_id']} | {r['category_id']} | "
            f"{r['rank']} | {r['score']:.3f} | {iou} | `{r['overlay']}` |"
        )
    md.append("")
    md.append("## How to read")
    md.append("")
    md.append("- Colored fill + contour = prediction; **white** contour = GT `mask_visib` (when present).")
    md.append("- `IoU(gt)` is best-match over GT instances of that category on the image.")
    md.append("- Source names are provenance tags (`cnos` / `muse` / `muse-repro` / …); do not relabel.")
    md.append("")
    (out / "INDEX.md").write_text("\n".join(md) + "\n")
    print(f"done: {len(all_rows)} panels → {out}", flush=True)


if __name__ == "__main__":
    main()
