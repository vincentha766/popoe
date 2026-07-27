#!/usr/bin/env python3
"""A/B: nearest-diameter / soft size selection for confusable clamps (obj19/20).

Follow-up to ``eval_cnos_size_gate_ab.py``. The band gate is one-sided (rejects
too-big, not too-small). This script compares:

  v2              appearance top-1
  v3_band         extent ∈ [0.25, 1.1]×diam, then appearance top-1
  v3_band_fb      v3_band with appearance fallback if empty
  soft            appearance × Gaussian(log extent/diam) affinity top-1
  nearest         keep masks whose nearest CAD in the confusable set is the
                  query, then appearance top-1
  nearest_fb      nearest with appearance fallback if empty
  oracle          best IoU among candidates (selection ceiling)

Pools: ``own`` (label==query) and ``pool`` (labels in confusable pair).

Report **per scene** — frames are multi-view orbits, not i.i.d.

Example:
  uv run python scripts/eval_cnos_size_select_ab.py \\
      --out-dir outputs/cnos_size_select_ab
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from popoe.interfaces import ObjectModel, Scene
from popoe.segmentor_cnos_lab import (
    DepthSizeGate,
    DiameterSizeModel,
    select_by_nearest_diameter,
    select_by_soft_affinity,
)
from popoe.segmentor_detections import decode_detection_mask, load_bop_detections

DEFAULT_PAIR = (19, 20)
DEFAULT_IOU_THR = 0.5
METHODS = (
    "v2",
    "v3_band",
    "v3_band_fb",
    "soft",
    "nearest",
    "nearest_fb",
    "oracle",
)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def _load_diameters_m(models_info: Path) -> dict[int, float]:
    raw = json.loads(models_info.read_text())
    return {int(k): float(v["diameter"]) / 1000.0 for k, v in raw.items()}


def _index_gt(bop: Path, scene_ids: Iterable[int]) -> dict[tuple[int, int, int], int]:
    out: dict[tuple[int, int, int], int] = {}
    for sid in scene_ids:
        gt = json.loads((bop / "test" / f"{sid:06d}" / "scene_gt.json").read_text())
        for im_str, ents in gt.items():
            iid = int(im_str)
            for gi, e in enumerate(ents):
                out.setdefault((sid, iid, int(e["obj_id"])), gi)
    return out


def _load_depth_K(bop: Path, scene_id: int, im_id: int):
    sd = bop / "test" / f"{scene_id:06d}"
    cam = json.loads((sd / "scene_camera.json").read_text())[str(im_id)]
    K = np.asarray(cam["cam_K"], dtype=np.float64).reshape(3, 3)
    depth = (
        cv2.imread(str(sd / "depth" / f"{im_id:06d}.png"), cv2.IMREAD_UNCHANGED)
        .astype(np.float32)
        * float(cam["depth_scale"])
        / 1000.0
    )
    return depth, K


def _load_gt_mask(bop: Path, scene_id: int, im_id: int, gt_idx: int) -> np.ndarray:
    path = (
        bop / "test" / f"{scene_id:06d}" / "mask_visib"
        / f"{im_id:06d}_{gt_idx:06d}.png"
    )
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED) > 0


def _prepare_cands(
    cands: list[dict],
    scene: Scene,
    gate: DepthSizeGate,
) -> tuple[list[dict], list[np.ndarray], list[float], list[Optional[float]]]:
    """Decode masks once; compute extents with the gate's extent estimator."""
    dets, masks, apps, exts = [], [], [], []
    for d in cands:
        m = decode_detection_mask(d["segmentation"])
        dets.append(d)
        masks.append(m)
        apps.append(float(d["score"]))
        if scene.depth is None or int(m.sum()) < gate.min_pixels:
            exts.append(None)
        else:
            exts.append(gate.extent_3d(m, scene.depth, scene.K))
    return dets, masks, apps, exts


def _pick(
    method: str,
    *,
    dets: list[dict],
    masks: list[np.ndarray],
    apps: list[float],
    exts: list[Optional[float]],
    scene: Scene,
    obj: ObjectModel,
    pair_diams: dict[int, float],
    gate: DepthSizeGate,
    size_model: DiameterSizeModel,
) -> tuple[Optional[int], bool, Optional[float]]:
    """Return (index, empty, extent_of_pick)."""
    n = len(dets)
    if n == 0:
        return None, True, None

    if method == "v2":
        i = max(range(n), key=lambda j: apps[j])
        return i, False, exts[i]

    if method in ("v3_band", "v3_band_fb"):
        kept = []
        for j in range(n):
            ok, ext = gate.accepts(scene, obj, masks[j])
            if ok:
                kept.append((j, apps[j], ext))
        if kept:
            j, _, ext = max(kept, key=lambda t: t[1])
            return j, False, ext
        if method == "v3_band_fb":
            i = max(range(n), key=lambda j: apps[j])
            return i, False, exts[i]
        return None, True, None

    if method == "soft":
        rivals = [pair_diams[p] for p in pair_diams if p != int(obj.obj_id)]
        res = select_by_soft_affinity(
            apps, exts, float(obj.diameter), size_model,
            missing_extent_affinity=0.0,
            rival_diameters=rivals,
        )
        if res is None:
            return None, True, None
        return res.index, False, res.extent

    if method in ("nearest", "nearest_fb"):
        res = select_by_nearest_diameter(
            apps, exts, int(obj.obj_id), pair_diams, size_model,
            fallback_appearance=(method == "nearest_fb"),
        )
        if res is None:
            return None, True, None
        return res.index, False, res.extent

    if method == "oracle":
        best_i, best_iou = None, -1.0
        # oracle filled by caller with GT; here unused
        return best_i, True, None

    raise ValueError(method)


def _summarize(rows: list[dict], iou_thr: float) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["scene_id"], r["obj_id"], r["method"], r["pool"])].append(r)
    summary = []
    for (sid, oid, method, pool), rs in sorted(groups.items()):
        n = len(rs)
        summary.append({
            "scene_id": sid,
            "obj_id": oid,
            "method": method,
            "pool": pool,
            "n": n,
            "empty_rate": sum(r["empty"] for r in rs) / n,
            "correct_at_iou": sum(r["correct"] for r in rs) / n,
            "mean_iou": float(np.mean([r["iou"] for r in rs])),
            "median_iou": float(np.median([r["iou"] for r in rs])),
            "swap_rate": sum(r["swap"] for r in rs) / n,
            "iou_thr": iou_thr,
        })
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bop", type=Path, default=Path("bop_data/ycbv"))
    ap.add_argument(
        "--detections", type=Path,
        default=Path("data/detections/cnos/cnos-fastsam_ycbv-test.json"),
    )
    ap.add_argument("--targets", type=Path, default=None)
    ap.add_argument("--obj-ids", type=str, default="19,20")
    ap.add_argument("--pair", type=str, default="19,20",
                    help="confusable pair for nearest assignment")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/cnos_size_select_ab"))
    ap.add_argument("--iou-thr", type=float, default=DEFAULT_IOU_THR)
    ap.add_argument("--min-pixels", type=int, default=100)
    ap.add_argument("--sigma-log", type=float, default=0.20)
    ap.add_argument("--min-extent-ratio", type=float, default=0.25)
    ap.add_argument("--max-extent-ratio", type=float, default=1.1)
    args = ap.parse_args(argv)

    bop = args.bop
    targets_path = args.targets or (bop / "test_targets_bop19.json")
    obj_ids = {int(x) for x in args.obj_ids.split(",") if x.strip()}
    pair = tuple(int(x) for x in args.pair.split(",") if x.strip())
    if len(pair) < 2:
        print("--pair needs at least two obj ids", file=sys.stderr)
        return 2

    diameters = _load_diameters_m(bop / "models" / "models_info.json")
    pair_diams = {oid: diameters[oid] for oid in pair}
    targets = [
        t for t in json.loads(targets_path.read_text())
        if int(t["obj_id"]) in obj_ids
    ]
    scene_ids = sorted({int(t["scene_id"]) for t in targets})
    gt_index = _index_gt(bop, scene_ids)

    print(f"loading detections from {args.detections} …", flush=True)
    all_dets = load_bop_detections(str(args.detections), source="cnos")
    by_img: dict[tuple[int, int], list[dict]] = defaultdict(list)
    pair_set = set(pair)
    for d in all_dets:
        if int(d["category_id"]) in pair_set:
            by_img[(int(d["scene_id"]), int(d["image_id"]))].append(d)
    print(
        f"targets={len(targets)} scenes={scene_ids} "
        f"pair-dets={sum(len(v) for v in by_img.values())}",
        flush=True,
    )

    gate = DepthSizeGate(
        min_extent_ratio=args.min_extent_ratio,
        max_extent_ratio=args.max_extent_ratio,
        min_pixels=args.min_pixels,
    )
    size_model = DiameterSizeModel(sigma_log=args.sigma_log)
    depth_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    detail_rows: list[dict] = []

    for ti, t in enumerate(targets):
        sid, iid, oid = int(t["scene_id"]), int(t["im_id"]), int(t["obj_id"])
        gt_idx = gt_index.get((sid, iid, oid))
        if gt_idx is None:
            print(f"skip missing GT s{sid} i{iid} obj{oid}", file=sys.stderr)
            continue
        if (sid, iid) not in depth_cache:
            depth_cache[(sid, iid)] = _load_depth_K(bop, sid, iid)
        depth, K = depth_cache[(sid, iid)]
        gt_mask = _load_gt_mask(bop, sid, iid, gt_idx)

        partner = next((p for p in pair if p != oid), None)
        partner_mask = None
        if partner is not None and (sid, iid, partner) in gt_index:
            partner_mask = _load_gt_mask(
                bop, sid, iid, gt_index[(sid, iid, partner)]
            )

        img_dets = by_img.get((sid, iid), [])
        scene = Scene(
            rgb=np.zeros((*gt_mask.shape, 3), np.uint8),
            depth=depth, K=K, scene_id=sid, im_id=iid,
        )
        obj = ObjectModel(oid, f"obj_{oid:06d}", diameter=diameters[oid])

        for pool_name, cands in (
            ("own", [d for d in img_dets if int(d["category_id"]) == oid]),
            ("pool", list(img_dets)),
        ):
            dets, masks, apps, exts = _prepare_cands(cands, scene, gate)

            # oracle
            best_iou = 0.0
            for m in masks:
                best_iou = max(best_iou, _mask_iou(m, gt_mask))
            detail_rows.append({
                "scene_id": sid, "im_id": iid, "obj_id": oid,
                "method": "oracle", "pool": pool_name,
                "empty": len(masks) == 0,
                "iou": best_iou,
                "partner_iou": 0.0,
                "swap": False,
                "correct": best_iou >= args.iou_thr,
                "score": 0.0,
                "category_id": None,
                "extent": None,
                "n_cands": len(masks),
                "diameter_m": diameters[oid],
            })

            for method in METHODS:
                if method == "oracle":
                    continue
                idx, empty, ext = _pick(
                    method,
                    dets=dets, masks=masks, apps=apps, exts=exts,
                    scene=scene, obj=obj, pair_diams=pair_diams,
                    gate=gate, size_model=size_model,
                )
                if empty or idx is None:
                    detail_rows.append({
                        "scene_id": sid, "im_id": iid, "obj_id": oid,
                        "method": method, "pool": pool_name,
                        "empty": True, "iou": 0.0, "partner_iou": 0.0,
                        "swap": False, "correct": False, "score": 0.0,
                        "category_id": None, "extent": None,
                        "n_cands": len(masks), "diameter_m": diameters[oid],
                    })
                    continue
                m = masks[idx]
                d = dets[idx]
                iou = _mask_iou(m, gt_mask)
                p_iou = (
                    _mask_iou(m, partner_mask) if partner_mask is not None else 0.0
                )
                swap = (
                    partner_mask is not None
                    and p_iou > iou
                    and p_iou >= args.iou_thr
                )
                detail_rows.append({
                    "scene_id": sid, "im_id": iid, "obj_id": oid,
                    "method": method, "pool": pool_name,
                    "empty": False,
                    "iou": iou,
                    "partner_iou": p_iou,
                    "swap": swap,
                    "correct": iou >= args.iou_thr,
                    "score": float(d["score"]),
                    "category_id": int(d["category_id"]),
                    "extent": ext,
                    "n_cands": len(masks),
                    "diameter_m": diameters[oid],
                })

        if (ti + 1) % 50 == 0 or ti + 1 == len(targets):
            print(f"  {ti + 1}/{len(targets)} targets", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / "detail.csv"
    summary_path = args.out_dir / "summary_per_scene.csv"
    meta_path = args.out_dir / "meta.json"

    with detail_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)

    summary = _summarize(detail_rows, args.iou_thr)
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    meta = {
        "bop": str(bop),
        "detections": str(args.detections),
        "obj_ids": sorted(obj_ids),
        "pair": list(pair),
        "n_targets": len(targets),
        "scenes": scene_ids,
        "iou_thr": args.iou_thr,
        "sigma_log": args.sigma_log,
        "gate": {
            "min_extent_ratio": args.min_extent_ratio,
            "max_extent_ratio": args.max_extent_ratio,
            "min_pixels": args.min_pixels,
        },
        "methods": {
            "v2": "appearance top-1",
            "v3_band": "ratio band then appearance",
            "v3_band_fb": "v3_band + appearance fallback",
            "soft": "appearance × softmax size share vs confusable diameters",
            "nearest": "nearest diameter among pair, then appearance",
            "nearest_fb": "nearest + appearance fallback",
            "oracle": "best IoU in pool",
        },
        "caveat": (
            "Per-scene reporting only; multi-view frames are correlated."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    smap = {(s["scene_id"], s["obj_id"], s["method"], s["pool"]): s for s in summary}
    print("\n=== correct@IoU (mean_iou) per scene / pool ===")
    hdr = (
        f"{'sc':>4} {'obj':>3} {'pool':>5} "
        f"{'v2':>6} {'band':>6} {'soft':>6} {'near':>6} {'nfb':>6} {'ora':>6} {'n':>3}"
    )
    print(hdr)
    for sid in scene_ids:
        for oid in sorted(obj_ids):
            for pool in ("own", "pool"):
                if (sid, oid, "v2", pool) not in smap:
                    continue
                def g(m):
                    s = smap.get((sid, oid, m, pool), {})
                    return s.get("correct_at_iou", 0.0)
                n = smap[(sid, oid, "v2", pool)]["n"]
                print(
                    f"{sid:4d} {oid:3d} {pool:>5} "
                    f"{g('v2'):6.3f} {g('v3_band'):6.3f} {g('soft'):6.3f} "
                    f"{g('nearest'):6.3f} {g('nearest_fb'):6.3f} {g('oracle'):6.3f} "
                    f"{n:3d}"
                )

    print("\n=== swap rate (pool) ===")
    print(f"{'sc':>4} {'obj':>3} {'v2':>6} {'band':>6} {'soft':>6} {'near':>6}")
    for sid in scene_ids:
        for oid in sorted(obj_ids):
            if (sid, oid, "v2", "pool") not in smap:
                continue
            def gs(m):
                return smap.get((sid, oid, m, "pool"), {}).get("swap_rate", 0.0)
            print(
                f"{sid:4d} {oid:3d} "
                f"{gs('v2'):6.3f} {gs('v3_band'):6.3f} {gs('soft'):6.3f} "
                f"{gs('nearest'):6.3f}"
            )

    print(f"\nwrote {detail_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
