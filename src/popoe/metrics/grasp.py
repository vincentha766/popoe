"""Grasping-axis pose metrics from a BOP prediction CSV (no renderer needed).

Port of gedi `scripts/freezev2_grasp_eval.py` (2026-08-06), byte-compatible in
aggregation and output so numbers reconcile against the archived grasp rows in
`REPRODUCTION.md` (headline rows #5/#6). Per that ledger's calibre note, the
grasp axis deliberately keeps the archived script's statistics — **mean over
per-object recalls** and **median over per-object medians** — NOT the BOP flat
calibre used by metrics/ar.py and metrics/vsd.py. Do not "fix" this without
re-baselining the archived numbers.

Reports, per object then averaged:
  ADD(-S) recall @ 0.1d : fraction with ADD(-S) < 0.10 * diameter   <- robotics standard
  ADD(-S) recall @ 0.05d: tighter (precise grasp)
  median |t| error (mm) : translation error
  median rot error (deg): rotation error
ADD-S (closest-point) is used for symmetric objects, ADD for the rest
(YCB-Video convention). Unregistered/dummy predictions (identity pose) count
as failures via their large errors.

Usage:
  BOP_PATH=/workspace/bop_data/ycbv \
  python -m popoe.metrics.grasp preds.csv
Env: POPOE_BOP_TOOLKIT (default /workspace/bop_toolkit), BOP_PATH.
"""
import os, sys, csv, json
from pathlib import Path
import numpy as np


def _pose_error():
    sys.path.insert(0, os.environ.get("POPOE_BOP_TOOLKIT", "/workspace/bop_toolkit"))
    from bop_toolkit_lib import pose_error
    return pose_error


def load_gt(bop_path, scenes):
    """(scene_id, im_id, obj_id) -> list of {R, t} ground-truth poses."""
    gt = {}
    for s in scenes:
        sdir = bop_path / "test" / f"{s:06d}"
        scene_gt = json.load(open(sdir / "scene_gt.json"))
        for im_id_str, gts in scene_gt.items():
            for g in gts:
                gt.setdefault((s, int(im_id_str), g["obj_id"]), []).append(dict(
                    R=np.array(g["cam_R_m2c"]).reshape(3, 3),
                    t=np.array(g["cam_t_m2c"]).reshape(3, 1)))
    return gt


def load_models(bop_path):
    """obj_id -> {pts, diam, sym} from models_eval (BOP convention)."""
    import trimesh
    mi = json.load(open(bop_path / "models_eval" / "models_info.json"))
    obj = {}
    for k, v in mi.items():
        oid = int(k)
        m = trimesh.load(bop_path / "models_eval" / f"obj_{oid:06d}.ply", force="mesh")
        sym = ("symmetries_continuous" in v) or ("symmetries_discrete" in v)
        obj[oid] = dict(pts=np.array(m.vertices), diam=v["diameter"], sym=sym)
    return obj


def per_object_errors(rows, gt, obj):
    """obj_id -> list of (add_or_adi, te_mm, re_deg); best over matching GTs."""
    pose_error = None  # deferred: bop_toolkit only needed once a row matches GT
    err = {}
    for r in rows:
        key = (int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"]))
        if key not in gt:
            continue
        if pose_error is None:
            pose_error = _pose_error()
        oid = int(r["obj_id"]); d = obj[oid]
        Re = np.array([float(x) for x in r["R"].split()]).reshape(3, 3)
        te_ = np.array([float(x) for x in r["t"].split()]).reshape(3, 1)
        best = None
        for g in gt[key]:
            e = (pose_error.adi if d["sym"] else pose_error.add)(Re, te_, g["R"], g["t"], d["pts"])
            if best is None or e < best[0]:
                best = (e, pose_error.te(te_, g["t"]), pose_error.re(Re, g["R"]))
        err.setdefault(oid, []).append(best)
    return err


def aggregate_grasp(err, obj):
    """Archived-script calibre: mean of per-object recalls; median of medians."""
    per_object = {}
    r10, r05, tmed, rmed = [], [], [], []
    for oid in sorted(err):
        e = np.array(err[oid]); d = obj[oid]
        a = e[:, 0]
        rec10 = float((a < 0.10 * d["diam"]).mean())
        rec05 = float((a < 0.05 * d["diam"]).mean())
        mt = float(np.median(e[:, 1])); mr = float(np.median(e[:, 2]))
        per_object[oid] = dict(n=len(e), rec10=rec10, rec05=rec05,
                               med_t_mm=mt, med_r_deg=mr, sym=obj[oid]["sym"])
        r10.append(rec10); r05.append(rec05); tmed.append(mt); rmed.append(mr)
    summary = dict(
        add_s_recall_01d=float(np.mean(r10)),
        add_s_recall_005d=float(np.mean(r05)),
        median_t_mm=float(np.median(tmed)),
        median_r_deg=float(np.median(rmed)),
    )
    return per_object, summary


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    csv_path = argv[0]
    bop_path = Path(os.environ.get("BOP_PATH", "/workspace/bop_data/lmo"))

    rows = list(csv.DictReader(open(csv_path)))
    print(f"loaded {len(rows)} rows from {csv_path}", flush=True)

    scenes = sorted({int(r["scene_id"]) for r in rows})
    gt = load_gt(bop_path, scenes)
    obj = load_models(bop_path)
    err = per_object_errors(rows, gt, obj)
    per_object, summary = aggregate_grasp(err, obj)

    print(f"\n{'obj':<5}{'#':<5}{'ADD(-S)<.1d':<12}{'<.05d':<9}{'med_t(mm)':<11}{'med_r(deg)':<11}{'sym'}")
    for oid, p in per_object.items():
        print(f"{oid:<5}{p['n']:<5}{p['rec10']:<12.3f}{p['rec05']:<9.3f}"
              f"{p['med_t_mm']:<11.1f}{p['med_r_deg']:<11.1f}{'Y' if p['sym'] else ''}")

    print(f"\n=== {Path(csv_path).name}  (BOP={bop_path.name}) ===")
    print(f"ADD(-S) recall @0.1d  : {summary['add_s_recall_01d']:.4f}   <- grasping standard")
    print(f"ADD(-S) recall @0.05d : {summary['add_s_recall_005d']:.4f}")
    print(f"median translation    : {summary['median_t_mm']:.1f} mm")
    print(f"median rotation       : {summary['median_r_deg']:.1f} deg")
    return summary


if __name__ == "__main__":
    main()
