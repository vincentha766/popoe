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
    a(f"# faithful-4way 候选分割对比 — LM-O scene {args.scene}（{names}）\n")
    a("本文件由 `scripts/export_faithful4way_seg_review.py` 生成，"
      "所有数字来自脚本，未经手工转录。\n")

    a("## 这些候选是怎么来的\n")
    a("不是重跑 faithful-4way 得到的。候选集合是四个检测 JSON 加几个开关的"
      "确定性函数，所以本地 CPU 就能复现出 pose 阶段真正看到的那一批 mask。"
      "脚本调用的是 `examples/bop_eval.py` 用的同一个 "
      "`BOPDetectionsSegmentor`，参数一一对应：\n")
    a("| bop_eval.py 的开关 | 传给 segmentor 的值 | 作用 |")
    a("|---|---|---|")
    a("| `--sources cnos=…,sam6d=…,nids=…,muse=…` | `sources={4 个}` | "
      "四源并集 |")
    a("| `--min-mask-pixels 0` | `min_pixels=0` | 面积过滤全关 |")
    a("| `--mask-iou-dedupe 1.1` | `iou_dedupe=1.1` | "
      "重叠去重全关（IoU 不可能超过 1.0）；跨源本来就不去重 |")
    a("| `--merge none` | `merge_labels=None` | 不做标签合池 |")
    a(f"| `--topk {args.topk} --mask-m {args.mask_m}` | "
      f"`topk=max({args.topk}, "
      f"{'2N' if args.mask_m == '2n' else 'N+1'})` 每 (源, 标签) 桶 | "
      "每源的候选上限 |")
    a("")
    caps = sorted({r["cap"] for r in records})
    insts = sorted({r["inst_count"] for r in records})
    a(f"LM-O 的 BOP19 target 全部 `inst_count={insts[0]}`，"
      f"所以本文件涉及的每个 target 上限都是 **{caps[0]} 个/源**，"
      f"一帧最多 4 × {caps[0]} = **{4 * caps[0]} 个候选**。"
      "跨源重复是故意保留的（FreeZe 的 top-M union without filtering），"
      "所以四个源看同一个物体时会出现四个几乎一样的 mask。\n")

    a("## 颜色对照\n")
    a("| 颜色 | 含义 |")
    a("|---|---|")
    a(f"| 品红 `rgb{GT_COLOR}` 填充 | GT `mask_visib`（仅 GT 行） |")
    a(f"| 白色 1px 轮廓 / 白框 | GT 轮廓与 GT 包围框，画在每一行上用来对位 |")
    for s, c in SRC_COLOR.items():
        a(f"| `rgb{c}` 填充+轮廓 | **{s}** 的候选 mask |")
    a("")
    a("overlay 列裁剪窗口固定按 GT 包围框计算（padding = GT 长边的 0.6 倍），"
      "同一张表的每一行共用一个窗口，可以直接横向比。"
      "小目标按 NEAREST 放大到长边 "
      f"{args.thumb} px——不用 LANCZOS，因为插值会把 1px 的边界误差糊掉，"
      "而这正是要看的东西。整帧列不裁剪，用来看跑偏的 mask 落在哪里。\n")

    a(f"## 全 split 统计（scene {args.scene} 的全部 BOP19 target）\n")
    a("三帧图是个例，这张表才是结论。命中 = 该候选与 GT `mask_visib` "
      f"的 IoU ≥ {HIT}。\n")
    a("| 物体 | target 数 | 平均候选数 | 至少一个候选命中 | 平均最佳 IoU | "
      "检测分最高的那个候选命中 |")
    a("|---|---|---|---|---|---|")
    for o in objs:
        v = agg[o]
        a(f"| **{o} {_name(o)}** | {v['n_targets']} | {v['mean_cands']:.2f} | "
          f"{v['hit_rate']:.3f} | {v['mean_best_iou']:.3f} | "
          f"{v['top1_hit_rate']:.3f} |")
    a("")
    a("拆到每个源（平均候选数上限为 "
      f"{caps[0]}；空手率 = 该源在这个物体上一个候选都不给的帧的占比）：\n")
    head = "| 物体 | 指标 | " + " | ".join(SRC_COLOR) + " |"
    a(head)
    a("|---|---|" + "---|" * len(SRC_COLOR))
    for o in objs:
        ps = agg[o]["per_source"]
        a(f"| **{o} {_name(o)}** | 平均候选数 | "
          + " | ".join(f"{ps[s]['mean_cands']:.2f}" for s in SRC_COLOR) + " |")
        a(f"| | 空手率 | "
          + " | ".join(f"{ps[s]['empty_rate']:.3f}" for s in SRC_COLOR) + " |")
        a(f"| | 命中率 | "
          + " | ".join(f"{ps[s]['hit_rate']:.3f}" for s in SRC_COLOR) + " |")
    a("")

    a("## 选了哪三帧，怎么选的\n")
    a("在两个物体同时是 BOP19 target 的帧里，按**较难那个物体的最佳候选 "
      "IoU** 排序，取第 10 / 50 / 90 百分位三帧；另一个物体用**同样三帧**"
      "渲染。同一张图、同样的光照和遮挡，差异才能归到物体本身。"
      f"本次的三帧是 {', '.join(str(f) for f in frames)}。\n")

    for r in records:
        o, im = r["obj_id"], r["im_id"]
        vis = ", ".join(f"{v:.3f}" for v in r["visib"]) or "—"
        nhit = sum(c["iou"] >= HIT for c in r["cands"])
        a(f"### image {im} · obj {o} {_name(o)}\n")
        a(f"GT 可见 {r['gt_area']} px，`visib_fract` {vis}，"
          f"`inst_count` {r['inst_count']} → 上限 {r['cap']} 个/源。"
          f"实得 **{len(r['cands'])} 个候选，其中 {nhit} 个命中**"
          f"（IoU ≥ {HIT}）。\n")
        a("| # | 来源 | 检测分 | 面积 px | IoU vs GT | overlay（GT 邻域） | "
          "整帧位置 |")
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
            a("> 四个源在这一帧对这个物体**一个候选都没有给**。"
              "pose 阶段收到空列表，这个 target 直接以零分记账——"
              "不是姿态估计错了，是它从来没拿到过输入。")
        a("")

    a("## 提供检测输入的四个文件\n")
    a("| 源 | 文件 |")
    a("|---|---|")
    for s, p in sources.items():
        a(f"| {s} | `{p}` |")
    a("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
