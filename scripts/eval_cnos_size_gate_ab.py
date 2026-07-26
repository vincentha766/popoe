#!/usr/bin/env python3
"""A/B: CNOS mask selection with vs without depth size gate (obj19/20 clamps).

Validates the real-scene cnos_match3 geometry prior on BOP YCB-V, using *existing*
official CNOS-FastSAM detections (no SAM/DINO re-run):

  v2  — pick top-1 by detector appearance score
  v3  — keep only masks whose 3D extent is in [0.25, 1.1] × query diameter,
        then pick top-1 by score among survivors

Candidate pools:
  own  — category_id == query object
  pool — category_id ∈ {19, 20}  (confusable same-shape pair; ceiling check
         showed obj20's true mask often sits under the obj19 label)

Metrics are reported **per scene** (frames within a scene are highly correlated;
do not treat the 300 targets as i.i.d.). IoU is vs BOP ``mask_visib``.

Example:
  uv run python scripts/eval_cnos_size_gate_ab.py \\
      --bop bop_data/ycbv \\
      --detections data/detections/cnos/cnos-fastsam_ycbv-test.json \\
      --out-dir outputs/cnos_size_gate_ab
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
from popoe.segmentor_cnos_v3 import DepthSizeGate
from popoe.segmentor_detections import decode_detection_mask, load_bop_detections

CLAMP_PAIR = (19, 20)
DEFAULT_IOU_THR = 0.5


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def _load_diameters_m(models_info: Path) -> dict[int, float]:
    raw = json.loads(models_info.read_text())
    # BOP models_info diameter is millimetres.
    return {int(k): float(v["diameter"]) / 1000.0 for k, v in raw.items()}


def _index_gt(bop: Path, scene_ids: Iterable[int]) -> dict[tuple[int, int, int], int]:
    """Map (scene_id, im_id, obj_id) -> first gt_idx with that obj_id."""
    out: dict[tuple[int, int, int], int] = {}
    for sid in scene_ids:
        gt = json.loads((bop / "test" / f"{sid:06d}" / "scene_gt.json").read_text())
        for im_str, ents in gt.items():
            iid = int(im_str)
            for gi, e in enumerate(ents):
                key = (sid, iid, int(e["obj_id"]))
                # first occurrence wins (YCB-V targets have one instance per obj)
                out.setdefault(key, gi)
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
        bop
        / "test"
        / f"{scene_id:06d}"
        / "mask_visib"
        / f"{im_id:06d}_{gt_idx:06d}.png"
    )
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED) > 0


def _pick_v2(cands: list[dict]) -> tuple[Optional[dict], Optional[np.ndarray]]:
    if not cands:
        return None, None
    best = max(cands, key=lambda d: d["score"])
    return best, decode_detection_mask(best["segmentation"])


def _pick_v3(
    cands: list[dict],
    *,
    scene: Scene,
    obj: ObjectModel,
    gate: DepthSizeGate,
    fallback: bool,
) -> tuple[Optional[dict], Optional[np.ndarray], int, Optional[float]]:
    kept: list[tuple[dict, np.ndarray, float]] = []
    for d in cands:
        m = decode_detection_mask(d["segmentation"])
        ok, ext = gate.accepts(scene, obj, m)
        if ok and ext is not None:
            kept.append((d, m, float(ext)))
    if kept:
        d, m, ext = max(kept, key=lambda t: t[0]["score"])
        return d, m, len(kept), ext
    if fallback:
        d, m = _pick_v2(cands)
        return d, m, 0, None
    return None, None, 0, None


def _summarize(rows: list[dict], iou_thr: float = DEFAULT_IOU_THR) -> list[dict]:
    """Per (scene_id, obj_id, method, pool) aggregates."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["scene_id"], r["obj_id"], r["method"], r["pool"])
        groups[key].append(r)

    summary = []
    for (sid, oid, method, pool), rs in sorted(groups.items()):
        n = len(rs)
        n_empty = sum(1 for r in rs if r["empty"])
        n_ok = sum(1 for r in rs if r["correct"])
        ious = [r["iou"] for r in rs]
        swaps = [r for r in rs if r.get("swap")]
        summary.append(
            {
                "scene_id": sid,
                "obj_id": oid,
                "method": method,
                "pool": pool,
                "n": n,
                "empty_rate": n_empty / n if n else 0.0,
                "correct_at_iou": n_ok / n if n else 0.0,
                "mean_iou": float(np.mean(ious)) if ious else 0.0,
                "median_iou": float(np.median(ious)) if ious else 0.0,
                "swap_rate": (len(swaps) / n) if n else 0.0,
                "iou_thr": iou_thr,
            }
        )
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bop", type=Path, default=Path("bop_data/ycbv"))
    ap.add_argument(
        "--detections",
        type=Path,
        default=Path("data/detections/cnos/cnos-fastsam_ycbv-test.json"),
    )
    ap.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="defaults to <bop>/test_targets_bop19.json",
    )
    ap.add_argument("--obj-ids", type=str, default="19,20")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/cnos_size_gate_ab"))
    ap.add_argument("--iou-thr", type=float, default=DEFAULT_IOU_THR)
    ap.add_argument(
        "--min-pixels",
        type=int,
        default=100,
        help="DepthSizeGate min_pixels (BOP 640×480: 100; real-scene match3 used 8000)",
    )
    ap.add_argument("--min-extent-ratio", type=float, default=0.25)
    ap.add_argument("--max-extent-ratio", type=float, default=1.1)
    args = ap.parse_args(argv)

    bop: Path = args.bop
    targets_path = args.targets or (bop / "test_targets_bop19.json")
    obj_ids = {int(x) for x in args.obj_ids.split(",") if x.strip()}
    if not obj_ids <= set(CLAMP_PAIR):
        print(
            f"warning: designed for clamp pair {CLAMP_PAIR}; got {sorted(obj_ids)}",
            file=sys.stderr,
        )

    diameters = _load_diameters_m(bop / "models" / "models_info.json")
    targets = [
        t
        for t in json.loads(targets_path.read_text())
        if int(t["obj_id"]) in obj_ids
    ]
    scene_ids = sorted({int(t["scene_id"]) for t in targets})
    gt_index = _index_gt(bop, scene_ids)

    print(f"loading detections from {args.detections} …", flush=True)
    all_dets = load_bop_detections(str(args.detections), source="cnos")
    by_img: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for d in all_dets:
        if int(d["category_id"]) in CLAMP_PAIR:
            by_img[(int(d["scene_id"]), int(d["image_id"]))].append(d)
    print(
        f"targets={len(targets)} scenes={scene_ids} "
        f"clamp-dets={sum(len(v) for v in by_img.values())}",
        flush=True,
    )

    gate = DepthSizeGate(
        min_extent_ratio=args.min_extent_ratio,
        max_extent_ratio=args.max_extent_ratio,
        min_pixels=args.min_pixels,
    )

    # cache depth/K and partner masks per image
    depth_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    detail_rows: list[dict] = []

    for ti, t in enumerate(targets):
        sid = int(t["scene_id"])
        iid = int(t["im_id"])
        oid = int(t["obj_id"])
        gt_idx = gt_index.get((sid, iid, oid))
        if gt_idx is None:
            print(f"skip missing GT s{sid} i{iid} obj{oid}", file=sys.stderr)
            continue

        if (sid, iid) not in depth_cache:
            depth_cache[(sid, iid)] = _load_depth_K(bop, sid, iid)
        depth, K = depth_cache[(sid, iid)]
        gt_mask = _load_gt_mask(bop, sid, iid, gt_idx)

        partner = 20 if oid == 19 else 19 if oid == 20 else None
        partner_mask = None
        if partner is not None and (sid, iid, partner) in gt_index:
            partner_mask = _load_gt_mask(
                bop, sid, iid, gt_index[(sid, iid, partner)]
            )

        img_dets = by_img.get((sid, iid), [])
        cands_own = [d for d in img_dets if int(d["category_id"]) == oid]
        cands_pool = list(img_dets)

        scene = Scene(
            rgb=np.zeros((*gt_mask.shape, 3), np.uint8),
            depth=depth,
            K=K,
            scene_id=sid,
            im_id=iid,
        )
        obj = ObjectModel(oid, f"obj_{oid:06d}", diameter=diameters[oid])

        # evaluate with real masks so we can mark swaps
        for pool_name, cands in (("own", cands_own), ("pool", cands_pool)):
            variants = [
                ("v2", False),
                ("v3", False),
                ("v3_fallback", True),
            ]
            for method, fallback in variants:
                if method == "v2":
                    d, m = _pick_v2(cands)
                    n_kept = len(cands)
                    ext = None
                else:
                    d, m, n_kept, ext = _pick_v3(
                        cands,
                        scene=scene,
                        obj=obj,
                        gate=gate,
                        fallback=fallback,
                    )
                empty = d is None or m is None
                iou = 0.0 if empty else _mask_iou(m, gt_mask)
                partner_iou = (
                    0.0
                    if empty or partner_mask is None
                    else _mask_iou(m, partner_mask)
                )
                swap = (
                    (not empty)
                    and partner_mask is not None
                    and partner_iou > iou
                    and partner_iou >= args.iou_thr
                )
                detail_rows.append(
                    {
                        "scene_id": sid,
                        "im_id": iid,
                        "obj_id": oid,
                        "method": method,
                        "pool": pool_name,
                        "empty": empty,
                        "iou": iou,
                        "partner_iou": partner_iou,
                        "swap": swap,
                        "correct": (not empty) and iou >= args.iou_thr,
                        "score": 0.0 if empty else float(d["score"]),
                        "category_id": None if empty else int(d["category_id"]),
                        "extent": ext,
                        "n_cands": len(cands),
                        "n_kept": n_kept,
                        "diameter_m": diameters[oid],
                    }
                )

            # oracle ceiling
            best_iou = 0.0
            for d in cands:
                m = decode_detection_mask(d["segmentation"])
                best_iou = max(best_iou, _mask_iou(m, gt_mask))
            detail_rows.append(
                {
                    "scene_id": sid,
                    "im_id": iid,
                    "obj_id": oid,
                    "method": "oracle",
                    "pool": pool_name,
                    "empty": len(cands) == 0,
                    "iou": best_iou,
                    "partner_iou": 0.0,
                    "swap": False,
                    "correct": best_iou >= args.iou_thr,
                    "score": 0.0,
                    "category_id": None,
                    "extent": None,
                    "n_cands": len(cands),
                    "n_kept": len(cands),
                    "diameter_m": diameters[oid],
                }
            )

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

    summary = _summarize(detail_rows, iou_thr=args.iou_thr)
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    meta = {
        "bop": str(bop),
        "detections": str(args.detections),
        "targets": str(targets_path),
        "obj_ids": sorted(obj_ids),
        "n_targets": len(targets),
        "scenes": scene_ids,
        "iou_thr": args.iou_thr,
        "gate": {
            "min_extent_ratio": args.min_extent_ratio,
            "max_extent_ratio": args.max_extent_ratio,
            "min_pixels": args.min_pixels,
            "note": (
                "Extent gate matches cnos_match3 / DepthSizeGate; "
                "min_pixels default 100 for BOP resolution (match3 real-scene used 8000)."
            ),
        },
        "methods": {
            "v2": "top-1 CNOS score, no size filter",
            "v3": "size gate then top-1 score; empty if no survivors",
            "v3_fallback": "v3, fall back to v2 if gate empties pool",
            "oracle": "best IoU among candidates (selection ceiling)",
        },
        "pools": {
            "own": "category_id == query",
            "pool": "category_id in {19,20}",
        },
        "caveat": (
            "300 targets come from few physical scenes with multi-view frames; "
            "report per-scene rates, do not treat as 300 i.i.d. samples."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    # Pretty print the main comparison table
    print("\n=== Per-scene correct@IoU≥{:.2f} (mean_iou) ===".format(args.iou_thr))
    header = (
        f"{'scene':>6} {'obj':>4} {'pool':>5} "
        f"{'v2_acc':>7} {'v2_iou':>7} "
        f"{'v3_acc':>7} {'v3_iou':>7} {'v3_empty':>8} "
        f"{'v3f_acc':>8} {'ora_acc':>7} {'n':>4}"
    )
    print(header)
    # index summary for lookup
    smap = {
        (s["scene_id"], s["obj_id"], s["method"], s["pool"]): s for s in summary
    }
    for sid in scene_ids:
        for oid in sorted(obj_ids):
            for pool in ("own", "pool"):
                keys_exist = any(
                    (sid, oid, m, pool) in smap
                    for m in ("v2", "v3", "v3_fallback", "oracle")
                )
                if not keys_exist:
                    continue
                v2 = smap.get((sid, oid, "v2", pool), {})
                v3 = smap.get((sid, oid, "v3", pool), {})
                v3f = smap.get((sid, oid, "v3_fallback", pool), {})
                ora = smap.get((sid, oid, "oracle", pool), {})
                n = v2.get("n", 0)
                print(
                    f"{sid:6d} {oid:4d} {pool:>5} "
                    f"{v2.get('correct_at_iou', 0):7.3f} {v2.get('mean_iou', 0):7.3f} "
                    f"{v3.get('correct_at_iou', 0):7.3f} {v3.get('mean_iou', 0):7.3f} "
                    f"{v3.get('empty_rate', 0):8.3f} "
                    f"{v3f.get('correct_at_iou', 0):8.3f} "
                    f"{ora.get('correct_at_iou', 0):7.3f} {n:4d}"
                )

    print("\n=== Per-scene swap rate (selected mask matches partner better) ===")
    print(f"{'scene':>6} {'obj':>4} {'pool':>5} {'v2_swap':>8} {'v3_swap':>8} {'v3f_swap':>9}")
    for sid in scene_ids:
        for oid in sorted(obj_ids):
            for pool in ("own", "pool"):
                v2 = smap.get((sid, oid, "v2", pool))
                v3 = smap.get((sid, oid, "v3", pool))
                v3f = smap.get((sid, oid, "v3_fallback", pool))
                if not v2:
                    continue
                print(
                    f"{sid:6d} {oid:4d} {pool:>5} "
                    f"{v2['swap_rate']:8.3f} {v3['swap_rate']:8.3f} {v3f['swap_rate']:9.3f}"
                )

    print(f"\nwrote {detail_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
