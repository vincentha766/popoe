#!/usr/bin/env python3
"""Offline pose-level size A/B for YCB-V clamps (obj19/20) — zero GPU.

Uses an existing ``bop_eval --cand-csv`` dump (must include ``metric_fit``).
Compares selection rules and a dual-CAD assignment that treats the same
``cand`` index under both CAD models as one physical mask (valid when both
objects were segmented with the same merge pool / top-K ordering).

Methods (per BOP target = scene, im, obj):

  no_mf          argmax s_icp * s_feat_1  (scale-blind)
  with_mf        argmax s_icp * s_feat_1 * metric_fit  (formal size-aware)
  dual_assign    for each cand index present under BOTH obj ids of the pair,
                 assign the mask to the CAD with higher metric_fit (tie →
                 higher s_icp*s_feat_1); each object then picks its best
                 remaining cand under with_mf. Falls back to with_mf when the
                 partner object is absent from the dump for that image.

Metrics vs BOP GT (``scene_gt``): ADD(-S) @ 0.1d and median |t| error (mm).
Report **per scene** (multi-view frames are correlated).

Example:
  uv run python scripts/eval_dual_cad_metric_fit_ab.py \\
      --cands ../gedi/ycbv_local_data/union_scoring_20260716/popoe_ycbv_union2_cands.csv \\
      --bop bop_data/ycbv \\
      --out-dir outputs/dual_cad_metric_fit_ab
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from popoe.confusable_select import dual_assign_rows, score_from_row

PAIR = (19, 20)
IDN = " ".join(f"{v:.6f}" for v in np.eye(3).flatten())
ZT = "0.0 0.0 0.0"


def _parse_vec(s: str) -> np.ndarray:
    return np.fromstring(s, sep=" ", dtype=np.float64)


def _parse_R(s: str) -> np.ndarray:
    return _parse_vec(s).reshape(3, 3)


def _load_gt_index(bop: Path, scene_ids):
    """(scene, im, obj) -> (R 3x3, t_mm 3, diam_m, gt_idx)."""
    mi = json.loads((bop / "models" / "models_info.json").read_text())
    diams = {int(k): float(v["diameter"]) / 1000.0 for k, v in mi.items()}
    out = {}
    for sid in scene_ids:
        gt = json.loads((bop / "test" / f"{sid:06d}" / "scene_gt.json").read_text())
        for im_str, ents in gt.items():
            iid = int(im_str)
            for gi, e in enumerate(ents):
                oid = int(e["obj_id"])
                if oid not in PAIR:
                    continue
                key = (sid, iid, oid)
                if key in out:
                    continue
                R = np.asarray(e["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
                t = np.asarray(e["cam_t_m2c"], dtype=np.float64)  # mm
                out[key] = (R, t, diams[oid], gi)
    return out


def _load_model_pts_mm(bop: Path, obj_id: int, n: int = 500) -> np.ndarray:
    """Sample CAD vertices in millimetres (BOP mesh units)."""
    try:
        import open3d as o3d
    except ImportError as e:
        raise SystemExit("open3d required for ADD(-S)") from e
    path = str(bop / "models" / f"obj_{obj_id:06d}.ply")
    mesh = o3d.io.read_triangle_mesh(path)
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(pts) > n:
        rng = np.random.default_rng(obj_id)
        pts = pts[rng.choice(len(pts), n, replace=False)]
    return pts


def add_s(pts_mm: np.ndarray, R_est, t_est_mm, R_gt, t_gt_mm) -> float:
    """Mean closest-point distance (mm) after pose — ADD-S for symmetric-friendly."""
    est = pts_mm @ R_est.T + t_est_mm
    gt = pts_mm @ R_gt.T + t_gt_mm
    # brute-force NN (n≈500)
    d2 = ((est[:, None, :] - gt[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(d2.min(axis=1)).mean())


def pick_independent(rows_for_target: list[dict], use_mf: bool) -> Optional[dict]:
    if not rows_for_target:
        return None
    return max(
        rows_for_target,
        key=lambda r: score_from_row(r, use_metric_fit=use_mf),
    )


def pick_dual_assign(
    rows_by_obj: dict[int, list[dict]],
    query_obj: int,
    partner_obj: int,
) -> Optional[dict]:
    """Assign each shared cand index to the CAD with higher metric_fit."""
    q_rows = rows_by_obj.get(query_obj, [])
    p_rows = rows_by_obj.get(partner_obj, [])
    pick = dual_assign_rows(q_rows, p_rows, use_metric_fit=True)
    return pick.row_or_hyp if pick.cand is not None else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--bop", type=Path, default=Path("bop_data/ycbv"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/dual_cad_metric_fit_ab"))
    ap.add_argument("--objs", type=str, default="19,20")
    ap.add_argument("--add-thr-frac", type=float, default=0.1,
                    help="ADD(-S) success threshold as fraction of diameter")
    args = ap.parse_args(argv)

    obj_ids = {int(x) for x in args.objs.split(",") if x.strip()}
    print(f"loading {args.cands} …", flush=True)
    # pandas optional — use csv for fewer deps in minimal envs; prefer pandas if present
    try:
        import pandas as pd
        df = pd.read_csv(args.cands)
        need = {"scene_id", "im_id", "obj_id", "cand", "w", "s_icp", "s_feat_1",
                "metric_fit", "score", "R", "t"}
        missing = need - set(df.columns)
        if missing:
            raise SystemExit(f"cand csv missing columns: {sorted(missing)}")
        df = df[df.obj_id.isin(obj_ids)]
        records = df.to_dict("records")
    except ImportError:
        records = []
        with args.cands.open() as f:
            for r in csv.DictReader(f):
                if int(r["obj_id"]) in obj_ids:
                    records.append(r)

    if not records:
        raise SystemExit("no rows for requested obj ids")

    # group rows
    by_target: dict[tuple, list] = defaultdict(list)
    for r in records:
        key = (int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"]))
        by_target[key].append(r)

    scene_ids = sorted({k[0] for k in by_target})
    gt_index = _load_gt_index(args.bop, scene_ids)
    pts_cache = {oid: _load_model_pts_mm(args.bop, oid) for oid in obj_ids}

    # images that have both objects in the dump
    imgs = defaultdict(set)
    for s, i, o in by_target:
        imgs[(s, i)].add(o)

    methods = ("no_mf", "with_mf", "dual_assign")
    detail = []

    targets = sorted(by_target.keys())
    print(f"targets={len(targets)} scenes={scene_ids}", flush=True)

    for ti, (sid, iid, oid) in enumerate(targets):
        rows = by_target[(sid, iid, oid)]
        partner = next((p for p in PAIR if p != oid), None)
        rows_by_obj = {
            oid: rows,
        }
        if partner is not None:
            rows_by_obj[partner] = by_target.get((sid, iid, partner), [])

        picks = {
            "no_mf": pick_independent(rows, use_mf=False),
            "with_mf": pick_independent(rows, use_mf=True),
            "dual_assign": pick_dual_assign(rows_by_obj, oid, partner)
            if partner is not None else pick_independent(rows, use_mf=True),
        }

        gt = gt_index.get((sid, iid, oid))
        if gt is None:
            print(f"skip no GT s{sid} i{iid} o{oid}", file=sys.stderr)
            continue
        R_gt, t_gt_mm, diam_m, _ = gt
        thr_mm = args.add_thr_frac * diam_m * 1000.0
        pts = pts_cache[oid]

        for method, row in picks.items():
            if row is None:
                detail.append({
                    "scene_id": sid, "im_id": iid, "obj_id": oid,
                    "method": method, "empty": True,
                    "add_s_mm": 1e9, "success": False, "t_err_mm": 1e9,
                    "metric_fit": None, "cand": None, "w": None,
                })
                continue
            R = _parse_R(row["R"])
            t_mm = _parse_vec(row["t"])
            ads = add_s(pts, R, t_mm, R_gt, t_gt_mm)
            t_err = float(np.linalg.norm(t_mm - t_gt_mm))
            detail.append({
                "scene_id": sid, "im_id": iid, "obj_id": oid,
                "method": method, "empty": False,
                "add_s_mm": ads,
                "success": ads <= thr_mm,
                "t_err_mm": t_err,
                "metric_fit": float(row["metric_fit"]),
                "cand": int(row["cand"]),
                "w": float(row["w"]),
            })

        if (ti + 1) % 50 == 0 or ti + 1 == len(targets):
            print(f"  {ti + 1}/{len(targets)}", flush=True)

    # summarize per scene
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / "detail.csv"
    summary_path = args.out_dir / "summary_per_scene.csv"

    with detail_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        w.writeheader()
        w.writerows(detail)

    groups = defaultdict(list)
    for r in detail:
        groups[(r["scene_id"], r["obj_id"], r["method"])].append(r)

    summary = []
    for (sid, oid, method), rs in sorted(groups.items()):
        n = len(rs)
        succ = [r for r in rs if r["success"]]
        t_errs = [r["t_err_mm"] for r in rs if not r["empty"]]
        summary.append({
            "scene_id": sid,
            "obj_id": oid,
            "method": method,
            "n": n,
            "add_s_recall": len(succ) / n if n else 0.0,
            "median_t_err_mm": float(np.median(t_errs)) if t_errs else None,
            "mean_t_err_mm": float(np.mean(t_errs)) if t_errs else None,
            "flip_vs_with_mf": None,  # filled below
        })

    # flip counts vs with_mf
    by_key = {(r["scene_id"], r["im_id"], r["obj_id"], r["method"]): r for r in detail}
    flip_acc = defaultdict(lambda: [0, 0])  # (sid,oid,method) -> [n_flip, n]
    for sid, iid, oid in targets:
        base = by_key.get((sid, iid, oid, "with_mf"))
        if base is None:
            continue
        for method in methods:
            if method == "with_mf":
                continue
            cur = by_key.get((sid, iid, oid, method))
            if cur is None:
                continue
            flip_acc[(sid, oid, method)][1] += 1
            if (cur.get("cand"), cur.get("w")) != (base.get("cand"), base.get("w")):
                flip_acc[(sid, oid, method)][0] += 1

    for s in summary:
        key = (s["scene_id"], s["obj_id"], s["method"])
        if key in flip_acc and flip_acc[key][1]:
            s["flip_vs_with_mf"] = flip_acc[key][0] / flip_acc[key][1]
        elif s["method"] == "with_mf":
            s["flip_vs_with_mf"] = 0.0

    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    meta = {
        "cands": str(args.cands),
        "bop": str(args.bop),
        "objs": sorted(obj_ids),
        "add_thr_frac": args.add_thr_frac,
        "methods": {
            "no_mf": "argmax s_icp*s_feat_1",
            "with_mf": "argmax s_icp*s_feat_1*metric_fit (formal size-aware)",
            "dual_assign": (
                "per shared cand index, assign to CAD with higher metric_fit; "
                "then with_mf among assigned (fallback with_mf)"
            ),
        },
        "caveat": (
            "cand index alignment assumes both CADs saw the same ordered merge "
            "pool. dual_assign only differs from with_mf when the partner object "
            "is also in the dump for that image. Per-scene reporting only."
        ),
        "n_images_with_both": int(sum(1 for oset in imgs.values() if len(oset) >= 2)),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print("\n=== ADD(-S)@0.1d recall (median t_err mm) per scene ===")
    print(f"{'sc':>4} {'obj':>3} {'no_mf':>14} {'with_mf':>14} {'dual':>14} {'n':>4}")
    smap = {(s["scene_id"], s["obj_id"], s["method"]): s for s in summary}
    for sid in scene_ids:
        for oid in sorted(obj_ids):
            if (sid, oid, "with_mf") not in smap:
                continue
            def fmt(m):
                s = smap[(sid, oid, m)]
                return f"{s['add_s_recall']:.3f}({s['median_t_err_mm']:.1f})"
            n = smap[(sid, oid, "with_mf")]["n"]
            print(f"{sid:4d} {oid:3d} {fmt('no_mf'):>14} {fmt('with_mf'):>14} "
                  f"{fmt('dual_assign'):>14} {n:4d}")

    # Export BOP-format CSVs (one row per target) for offline AR / grasp tools.
    for method in methods:
        path = args.out_dir / f"bop_{method}.csv"
        with path.open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
            for sid, iid, oid in targets:
                rows = by_target[(sid, iid, oid)]
                partner = next((p for p in PAIR if p != oid), None)
                rows_by_obj = {oid: rows}
                if partner is not None:
                    rows_by_obj[partner] = by_target.get((sid, iid, partner), [])
                if method == "no_mf":
                    row = pick_independent(rows, use_mf=False)
                elif method == "with_mf":
                    row = pick_independent(rows, use_mf=True)
                else:
                    row = (
                        pick_dual_assign(rows_by_obj, oid, partner)
                        if partner is not None
                        else pick_independent(rows, use_mf=True)
                    )
                if row is None:
                    wr.writerow([sid, iid, oid, 0.0, IDN, ZT, "0.0"])
                else:
                    sc = score_from_row(row, use_metric_fit=(method != "no_mf"))
                    wr.writerow([
                        sid, iid, oid, f"{sc:.6f}",
                        row["R"], row["t"], "0.0",
                    ])
        print(f"wrote {path}")

    print(f"\nwrote {detail_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
