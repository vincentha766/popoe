#!/usr/bin/env python3
"""Render the faithful-4way candidate segmentation set for chosen LM-O/YCB-V
objects, and emit a markdown comparison table. CPU-only, no pose estimation.

WHY THIS EXISTS SEPARATELY FROM export_bop_seg_review.py
--------------------------------------------------------
`export_bop_seg_review.py` samples frames PER SOURCE and draws each source's
top-K independently — useful for eyeballing one detector. This script instead
reproduces the exact candidate set that `examples/bop_eval.py` hands to the
pose stage under the faithful-4way recipe, by calling the SAME
`BOPDetectionsSegmentor` with the same knobs:

    --sources cnos=…,sam6d=…,nids=…,muse=…
    --min-mask-pixels 0        -> min_pixels=0   (nothing dropped by area)
    --mask-iou-dedupe 1.1      -> iou_dedupe=1.1 (nothing dropped by overlap;
                                  IoU cannot exceed 1.0, so the per-source
                                  near-duplicate filter is disabled by design)
    --merge none               -> merge_labels=None (no label pooling)
    --topk 2 --mask-m 2n       -> cap = max(2, 2*inst_count) per (source,label)

On LM-O every BOP19 target has inst_count == 1, so the cap is 2 for every
target and a frame yields AT MOST 4 sources x 2 = 8 candidates. Cross-source
duplicates are kept on purpose (FreeZe's "top-M union without filtering").

So the rendered rows ARE the pipeline's candidates, in the pipeline's order
(global sort by detector score), not a re-derivation. IoU against
`mask_visib` is added here for review only; the pipeline never sees GT.

Usage:
  .venv/bin/python scripts/export_faithful4way_seg_review.py \
      --bop bop_data/lmo --dataset lmo --objs 12,6 \
      --frames 476,434,263 \
      --out-dir outputs/faithful4way_seg_lmo
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from popoe.interfaces import ObjectModel, Scene
from popoe.segmentor_detections import BOPDetectionsSegmentor

# Same legend as outputs/seg_proposal_review/REVIEW.md — the two artifacts are
# read side by side, so the source->colour mapping must not drift.
SRC_COLOR = {
    "cnos": (255, 64, 64),      # red
    "sam6d": (64, 220, 80),     # green
    "nids": (64, 128, 255),     # blue
    "muse": (255, 192, 64),     # yellow
}
GT_COLOR = (255, 0, 200)        # magenta
WHITE = (255, 255, 255)

LMO_NAMES = {1: "ape", 5: "can", 6: "cat", 8: "driller", 9: "duck",
             10: "eggbox", 11: "glue", 12: "holepuncher"}


def floored_topk(user_topk: int, inst_count: int, mask_m: str = "2n") -> int:
    """Mirror of examples/bop_eval.py:floored_topk. Copied rather than imported
    because bop_eval pulls in torch/nvdiffrast at import time and this script
    must stay CPU-and-laptop runnable; the printed cap in the emitted markdown
    is what makes a drift visible."""
    floor = 2 * inst_count if mask_m == "2n" else inst_count + 1
    return max(user_topk, floor)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def overlay(rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45,
            contour: bool = True) -> np.ndarray:
    """Semi-transparent fill + 1px contour.

    CHAIN_APPROX_NONE and width 1 are deliberate (popoe 16bdfb1): SIMPLE
    straightens concavities away from the true boundary and a width-2 stroke
    straddles the edge, which makes a pixel-accurate mask look like it misses
    the object — the exact judgement this review is supposed to support.
    """
    import cv2
    out = rgb.copy()
    m = mask.astype(bool)
    c = np.asarray(color, np.float32)
    out[m] = (out[m].astype(np.float32) * (1 - alpha) + c * alpha).astype(np.uint8)
    if contour:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, tuple(int(x) for x in color), 1)
    return out


def bbox_of(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_box(img: np.ndarray, box, color, width: int = 2) -> np.ndarray:
    import cv2
    if box is None:
        return img
    out = img.copy()
    x0, y0, x1, y1 = [int(v) for v in box]
    cv2.rectangle(out, (x0, y0), (x1, y1), tuple(int(c) for c in color), width)
    return out


def crop_box_around(gt_box, shape, pad_frac: float = 0.6):
    """Crop window centred on the GT box, padded by pad_frac of its long side.
    Fixed to GT (not to the candidate) so every row of a table shares one
    window and masks can be compared by eye without mental rescaling."""
    h, w = shape[:2]
    if gt_box is None:
        return 0, 0, w, h
    x0, y0, x1, y1 = gt_box
    pad = int(round(pad_frac * max(x1 - x0, y1 - y0))) + 4
    return (max(0, x0 - pad), max(0, y0 - pad),
            min(w, x1 + pad + 1), min(h, y1 + pad + 1))


def save(img: np.ndarray, path: Path, long_side: int | None = None) -> None:
    pil = Image.fromarray(img)
    if long_side:
        w, h = pil.size
        s = long_side / max(w, h)
        if s > 1.0:
            # NEAREST on upscale: mask edges stay pixel-honest. LANCZOS would
            # feather the boundary and hide 1px errors this review looks for.
            pil = pil.resize((round(w * s), round(h * s)), Image.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path)


@lru_cache(maxsize=8)
def _scene_gt(scene_dir: Path):
    """scene_gt.json for LM-O scene 2 is ~1 MB and every target of the split
    needs it; parsing it per target dominated the aggregate pass."""
    return (json.loads((scene_dir / "scene_gt.json").read_text()),
            json.loads((scene_dir / "scene_gt_info.json").read_text()))


def gt_instances(scene_dir: Path, im_id: int, obj_id: int):
    """Return [(gt_idx, mask_visib bool array, visib_fract), ...]."""
    gt, info = _scene_gt(scene_dir)
    out = []
    for gi, inst in enumerate(gt.get(str(im_id), [])):
        if int(inst["obj_id"]) != int(obj_id):
            continue
        p = scene_dir / "mask_visib" / f"{im_id:06d}_{gi:06d}.png"
        arr = np.asarray(Image.open(p))
        if arr.ndim == 3:
            arr = arr[..., 0]
        out.append((gi, arr > 0, float(info[str(im_id)][gi]["visib_fract"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop", required=True)
    ap.add_argument("--dataset", default="lmo")
    ap.add_argument("--det-root", default="data/detections")
    ap.add_argument("--scene", type=int, default=2)
    ap.add_argument("--objs", required=True, help="comma-separated obj ids")
    ap.add_argument("--frames", required=True, help="comma-separated im ids")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--mask-m", choices=["n1", "2n"], default="2n")
    ap.add_argument("--thumb", type=int, default=260,
                    help="min long side for the crop panels (upscaled NEAREST)")
    args = ap.parse_args()

    bop = Path(args.bop)
    det = Path(args.det_root)
    ds = args.dataset
    sources = {
        "cnos": det / "cnos" / f"cnos-fastsam_{ds}-test.json",
        "sam6d": det / "sam6d" / f"sam6d_official_{ds}.json",
        "nids": det / "nids" / f"nids_wa_sappe_{ds}.json",
        "muse": det / "muse" / f"muse-full_{ds}-test.json",
    }
    for name, p in sources.items():
        if not p.is_file():
            raise SystemExit(f"missing detection file for {name}: {p}")

    seg = BOPDetectionsSegmentor(
        sources={k: str(v) for k, v in sources.items()},
        topk=args.topk, merge_labels=None, iou_dedupe=1.1, min_pixels=0,
    )

    scene_dir = bop / "test" / f"{args.scene:06d}"
    targets = json.loads((bop / "test_targets_bop19.json").read_text())
    inst_of = {(t["im_id"], t["obj_id"]): t["inst_count"] for t in targets
               if t["scene_id"] == args.scene}

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "img"
    objs = [int(x) for x in args.objs.split(",")]
    frames = [int(x) for x in args.frames.split(",")]

    records = []
    for im_id in frames:
        rgb = np.asarray(Image.open(
            scene_dir / "rgb" / f"{im_id:06d}.png").convert("RGB"))
        for obj_id in objs:
            n = inst_of.get((im_id, obj_id))
            if n is None:
                raise SystemExit(
                    f"(im {im_id}, obj {obj_id}) is not a BOP19 target — it "
                    f"would never be segmented by the pipeline either")
            cap = floored_topk(args.topk, n, args.mask_m)
            scene = Scene(rgb=rgb, depth=None, K=None,
                          scene_id=args.scene, im_id=im_id)
            obj = ObjectModel(obj_id=obj_id, mesh_path="", diameter=0.1)
            dets = seg.segment(scene, obj, topk=cap)

            gts = gt_instances(scene_dir, im_id, obj_id)
            gt_union = (np.logical_or.reduce([g for _, g, _ in gts])
                        if gts else np.zeros(rgb.shape[:2], bool))
            gt_box = bbox_of(gt_union)
            cx0, cy0, cx1, cy1 = crop_box_around(gt_box, rgb.shape)

            stem = f"im{im_id:06d}_obj{obj_id:02d}"
            # GT reference panels
            gt_over = overlay(rgb, gt_union, GT_COLOR, alpha=0.5, contour=False)
            save(gt_over[cy0:cy1, cx0:cx1], img_dir / f"{stem}_GT_crop.png",
                 args.thumb)
            save(draw_box(gt_over, gt_box, WHITE, 2),
                 img_dir / f"{stem}_GT_full.png")

            cands = []
            for rank, d in enumerate(dets):
                color = SRC_COLOR.get(d.source, (200, 200, 200))
                best = max((mask_iou(d.mask, g) for _, g, _ in gts), default=0.0)
                cname = f"{stem}_c{rank}_{d.source}"
                # crop: candidate fill in source colour + GT contour in white
                crop = overlay(rgb, d.mask, color, alpha=0.45)
                # alpha=0 -> no fill, contour only: the white outline is GT,
                # so a candidate's error is readable as colour-outside-white.
                crop = overlay(crop, gt_union, WHITE, alpha=0.0)
                save(crop[cy0:cy1, cx0:cx1], img_dir / f"{cname}_crop.png",
                     args.thumb)
                # full frame: where did this mask actually land?
                full = overlay(rgb, d.mask, color, alpha=0.45)
                full = draw_box(full, gt_box, WHITE, 2)
                save(full, img_dir / f"{cname}_full.png")
                cands.append({
                    "rank": rank, "source": d.source, "score": float(d.score),
                    "area": int(d.mask.sum()), "iou": best,
                    "crop": f"img/{cname}_crop.png",
                    "full": f"img/{cname}_full.png",
                })
            records.append({
                "im_id": im_id, "obj_id": obj_id, "inst_count": n, "cap": cap,
                "gt_area": int(gt_union.sum()),
                "visib": [v for _, _, v in gts],
                "gt_crop": f"img/{stem}_GT_crop.png",
                "gt_full": f"img/{stem}_GT_full.png",
                "cands": cands,
            })

    (out_dir / "index.json").write_text(json.dumps(records, indent=1))

    agg = aggregate(seg, scene_dir, targets, objs, args)
    (out_dir / "stats.json").write_text(json.dumps(agg, indent=1))
    (out_dir / "REVIEW.md").write_text(
        emit_md(records, agg, objs, frames, args, sources))
    print(f"wrote {len(records)} (frame, object) records + REVIEW.md -> {out_dir}")
    return 0


# ── whole-split statistics ───────────────────────────────────────────────
#
# The three rendered frames are anecdotes; these numbers are the claim. They
# are computed over EVERY BOP19 target of the chosen objects with the same
# segmentor instance, so the tables cannot disagree with the panels.

HIT = 0.5   # IoU at which a candidate is called a hit (BOP AP convention)


def aggregate(seg, scene_dir: Path, targets, obj_ids, args) -> dict:
    out = {}
    for obj_id in obj_ids:
        rows = [t for t in targets
                if t["obj_id"] == obj_id and t["scene_id"] == args.scene]
        per_src = {s: {"cands": 0, "empty": 0, "hit": 0} for s in SRC_COLOR}
        n_cands = hits = 0
        best_sum = 0.0
        top1_hits = 0
        for t in rows:
            im_id, n = t["im_id"], t["inst_count"]
            scene = Scene(rgb=np.zeros((1, 1), np.uint8), depth=None, K=None,
                          scene_id=t["scene_id"], im_id=im_id)
            obj = ObjectModel(obj_id=obj_id, mesh_path="", diameter=0.1)
            dets = seg.segment(scene, obj,
                               topk=floored_topk(args.topk, n, args.mask_m))
            gts = [g for _, g, _ in gt_instances(scene_dir, im_id, obj_id)]
            ious = [max((mask_iou(d.mask, g) for g in gts), default=0.0)
                    for d in dets]
            n_cands += len(dets)
            best = max(ious, default=0.0)
            best_sum += best
            hits += best >= HIT
            top1_hits += bool(ious) and ious[0] >= HIT
            for s in per_src:
                mine = [i for d, i in zip(dets, ious) if d.source == s]
                per_src[s]["cands"] += len(mine)
                per_src[s]["empty"] += not mine
                per_src[s]["hit"] += max(mine, default=0.0) >= HIT
        k = len(rows)
        out[obj_id] = {
            "n_targets": k,
            "mean_cands": n_cands / k,
            "hit_rate": hits / k,
            "mean_best_iou": best_sum / k,
            "top1_hit_rate": top1_hits / k,
            "per_source": {s: {"mean_cands": v["cands"] / k,
                               "empty_rate": v["empty"] / k,
                               "hit_rate": v["hit"] / k}
                           for s, v in per_src.items()},
        }
    return out


# ── markdown ─────────────────────────────────────────────────────────────

def _name(obj_id: int) -> str:
    return LMO_NAMES.get(obj_id, str(obj_id))


def emit_md(records, agg, objs, frames, args, sources) -> str:
    L = []
    a = L.append
    names = " / ".join(f"obj {o} {_name(o)}" for o in objs)
    a(f"# faithful-4way candidate segmentation — LM-O scene {args.scene} ({names})\n")
    a("Generated by `scripts/export_faithful4way_seg_review.py`. "
      "All numbers come from the script; none are transcribed by hand.\n")

    a("## How these candidates are produced\n")
    a("This is not a re-run of the pose recipe. The candidate set is a "
      "deterministic function of the four detection JSONs and a few flags, "
      "so a CPU machine can reconstruct the masks the pose stage actually "
      "sees. The script calls the same `BOPDetectionsSegmentor` that "
      "`examples/bop_eval.py` uses, with matching knobs:\n")
    a("| bop_eval.py flag | value passed to the segmentor | effect |")
    a("|---|---|---|")
    a("| `--sources cnos=…,sam6d=…,nids=…,muse=…` | `sources={4}` | "
      "four-source union |")
    a("| `--min-mask-pixels 0` | `min_pixels=0` | no area filter |")
    a("| `--mask-iou-dedupe 1.1` | `iou_dedupe=1.1` | "
      "no overlap filter (IoU cannot exceed 1.0); cross-source never deduped |")
    a("| `--merge none` | `merge_labels=None` | no label pooling |")
    a(f"| `--topk {args.topk} --mask-m {args.mask_m}` | "
      f"`topk=max({args.topk}, "
      f"{'2N' if args.mask_m == '2n' else 'N+1'})` per (source, label) | "
      "per-source candidate cap |")
    a("")
    caps = sorted({r["cap"] for r in records})
    insts = sorted({r["inst_count"] for r in records})
    a(f"Every LM-O BOP19 target has `inst_count={insts[0]}`, "
      f"so every target in this file is capped at **{caps[0]} per source**, "
      f"at most 4 × {caps[0]} = **{4 * caps[0]} candidates** per frame. "
      "Cross-source duplicates are kept on purpose (FreeZe's top-M union "
      "without filtering), so four sources looking at the same object can "
      "produce four nearly identical masks.\n")

    a("## Colour key\n")
    a("| colour | meaning |")
    a("|---|---|")
    a(f"| magenta `rgb{GT_COLOR}` fill | GT `mask_visib` (GT row only) |")
    a(f"| white 1px contour / box | GT contour and bbox, drawn on every row for alignment |")
    for s, c in SRC_COLOR.items():
        a(f"| `rgb{c}` fill+contour | **{s}** candidate mask |")
    a("")
    a("The overlay crop is always the GT bbox (padding = 0.6 × GT long side); "
      "every row in a table shares that window so columns line up. "
      "Small objects are upscaled with NEAREST to long side "
      f"{args.thumb} px — not LANCZOS, which would blur the 1px boundary "
      "error this review is looking at. The full-frame column is uncropped, "
      "to show where a drifted mask landed.\n")

    a(f"## Full-split stats (all BOP19 targets of scene {args.scene})\n")
    a("The three frames are examples; this table is the result. A hit is "
      f"IoU ≥ {HIT} against GT `mask_visib`.\n")
    a("| object | #targets | mean candidates | ≥1 candidate hit | mean best IoU | "
      "top detector-score candidate hits |")
    a("|---|---|---|---|---|---|")
    for o in objs:
        v = agg[o]
        a(f"| **{o} {_name(o)}** | {v['n_targets']} | {v['mean_cands']:.2f} | "
          f"{v['hit_rate']:.3f} | {v['mean_best_iou']:.3f} | "
          f"{v['top1_hit_rate']:.3f} |")
    a("")
    a("Per source (mean candidates capped at "
      f"{caps[0]}; empty rate = fraction of frames where that source "
      "gave this object no candidate):\n")
    head = "| object | metric | " + " | ".join(SRC_COLOR) + " |"
    a(head)
    a("|---|---|" + "---|" * len(SRC_COLOR))
    for o in objs:
        ps = agg[o]["per_source"]
        a(f"| **{o} {_name(o)}** | mean candidates | "
          + " | ".join(f"{ps[s]['mean_cands']:.2f}" for s in SRC_COLOR) + " |")
        a(f"| | empty rate | "
          + " | ".join(f"{ps[s]['empty_rate']:.3f}" for s in SRC_COLOR) + " |")
        a(f"| | hit rate | "
          + " | ".join(f"{ps[s]['hit_rate']:.3f}" for s in SRC_COLOR) + " |")
    a("")

    a("## Which three frames, and why\n")
    a("Among frames where both objects are BOP19 targets, rank by the "
      "**harder object's best-candidate IoU** and take the 10 / 50 / 90 "
      "percentile frames; the other object is rendered on the **same three "
      "frames**. Same image, lighting, and occlusion, so differences attach "
      "to the object. "
      f"This file's three frames are {', '.join(str(f) for f in frames)}.\n")

    for r in records:
        o, im = r["obj_id"], r["im_id"]
        vis = ", ".join(f"{v:.3f}" for v in r["visib"]) or "—"
        nhit = sum(c["iou"] >= HIT for c in r["cands"])
        a(f"### image {im} · obj {o} {_name(o)}\n")
        a(f"GT visible {r['gt_area']} px, `visib_fract` {vis}, "
          f"`inst_count` {r['inst_count']} → cap {r['cap']} per source. "
          f"**{len(r['cands'])} candidates, {nhit} hits** "
          f"(IoU ≥ {HIT}).\n")
        a("| # | source | detector score | area px | IoU vs GT | overlay (GT neighbourhood) | "
          "full frame |")
        a("|---|---|---|---|---|---|---|")
        a(f"| **GT** | — | — | {r['gt_area']} | — | "
          f"![]({r['gt_crop']}) | ![]({r['gt_full']}) |")
        for c in r["cands"]:
            mark = "**" if c["iou"] >= HIT else ""
            a(f"| {c['rank']} | {mark}{c['source']}{mark} | "
              f"`{c['score']:.3f}` | {c['area']} | "
              f"{mark}`{c['iou']:.3f}`{mark} | "
              f"![]({c['crop']}) | ![]({c['full']}) |")
        if not r["cands"]:
            a("")
            a("> None of the four sources proposed this object on this frame. "
              "The pose stage receives an empty list and the target scores "
              "zero — not a pose error; it never received an input.")
        a("")

    a("## Detection input files\n")
    a("| source | file |")
    a("|---|---|")
    for s, p in sources.items():
        a(f"| {s} | `{p}` |")
    a("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
