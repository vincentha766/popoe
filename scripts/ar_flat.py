"""BOP AR at the OFFICIAL aggregation, plus popoe's per-object one, side by side.

BOP defines recall as the fraction of *annotated object instances* that are
correct, so every instance carries equal weight and an object contributes in
proportion to how many instances it has. popoe's `metrics/ar.py` and
`metrics/vsd.py` instead average per-object recalls with equal weight per
OBJECT, which under-reports by 0.5-0.9 pt on LM-O / YCB-V (verified against the
BOP server: flat agrees to 0.03 pt, per-object does not).

This script does not touch popoe's scorers — it recomputes the same errors and
reports BOTH aggregations, so a number can always be quoted with its calibre.

MSSD / MSPD are recomputed here from the pose CSV (bop_toolkit pose_error, same
calls popoe's ar.py makes). VSD is NOT re-rendered: popoe's vsd.py already
persists the per-row, per-tau error tensor next to the CSV as
`<csv>.vsd_errs.npz`, so re-aggregating it is exact and free.

Usage:
  python ar_flat.py <pose.csv> <bop_root> [label]
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, os.environ.get("POPOE_BOP_TOOLKIT",
                                  "/workspace/bop_toolkit"))
from bop_toolkit_lib import misc, pose_error  # noqa: E402

MSSD_THRS = np.arange(0.05, 0.51, 0.05)   # fraction of diameter
MSPD_THRS = np.arange(5, 51, 5)           # pixels (reference width 640)
VSD_THS = np.arange(0.05, 0.51, 0.05)


def load_models(bop_root: Path, obj_ids):
    info = json.load(open(bop_root / "models_eval" / "models_info.json"))
    out = {}
    for o in obj_ids:
        m = trimesh.load(bop_root / "models_eval" / f"obj_{o:06d}.ply",
                         force="mesh")
        out[o] = dict(pts=np.array(m.vertices), diameter=info[str(o)]["diameter"],
                      syms=misc.get_symmetry_transformations(info[str(o)], 0.01))
    return out


def load_gt(bop_root: Path, scenes):
    gt = {}
    for s in scenes:
        sdir = bop_root / "test" / f"{s:06d}"
        scene_gt = json.load(open(sdir / "scene_gt.json"))
        cams = json.load(open(sdir / "scene_camera.json"))
        for im_s, gts in scene_gt.items():
            K = np.array(cams[im_s]["cam_K"]).reshape(3, 3)
            for g in gts:
                gt.setdefault((s, int(im_s), g["obj_id"]), []).append(dict(
                    R=np.array(g["cam_R_m2c"]).reshape(3, 3),
                    t=np.array(g["cam_t_m2c"]).reshape(3, 1), K=K))
    return gt


def errors(csv_path, bop_root: Path):
    """Per-ROW (scene, im, obj, mssd_mm, mspd_px). Rows with no GT for their
    (scene, im, obj) are dropped, matching popoe's ar.py so the two aggregations
    below differ ONLY in how they average."""
    rows = list(csv.DictReader(open(csv_path)))
    keys = [(r["scene_id"], r["im_id"], r["obj_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate (scene,im,obj) rows — single-instance only")
    models = load_models(bop_root, sorted({int(r["obj_id"]) for r in rows}))
    gt = load_gt(bop_root, sorted({int(r["scene_id"]) for r in rows}))
    out = []
    for r in rows:
        k = (int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"]))
        if k not in gt:
            continue
        R = np.array([float(x) for x in r["R"].split()]).reshape(3, 3)
        t = np.array([float(x) for x in r["t"].split()]).reshape(3, 1)
        d = models[k[2]]
        mssd = min(pose_error.mssd(R, t, g["R"], g["t"], d["pts"], d["syms"])
                   for g in gt[k])
        mspd = min(pose_error.mspd(R, t, g["R"], g["t"], g["K"], d["pts"],
                                   d["syms"]) for g in gt[k])
        out.append((k[2], mssd, mspd, d["diameter"]))
    return out


def agg(errs):
    """(flat, per_object) AR for MSSD and MSPD."""
    obj = np.array([e[0] for e in errs])
    mssd = np.array([e[1] for e in errs])
    mspd = np.array([e[2] for e in errs])
    diam = np.array([e[3] for e in errs])
    # FLAT — one weight per annotated instance, which is BOP's definition.
    f_mssd = np.mean([(mssd < th * diam).mean() for th in MSSD_THRS])
    f_mspd = np.mean([(mspd < th).mean() for th in MSPD_THRS])
    # PER-OBJECT — popoe's ar.py: recall per object, then equal-weight mean.
    p_mssd, p_mspd = [], []
    for o in np.unique(obj):
        m = obj == o
        p_mssd.append(np.mean([(mssd[m] < th * diam[m]).mean()
                               for th in MSSD_THRS]))
        p_mspd.append(np.mean([(mspd[m] < th).mean() for th in MSPD_THRS]))
    return (float(f_mssd), float(f_mspd),
            float(np.mean(p_mssd)), float(np.mean(p_mspd)), len(errs))


def vsd_from_npz(npz_path):
    """Re-aggregate popoe's persisted per-row per-tau VSD errors. Returns
    (flat, per_object) AR_VSD, or (None, None) if the file is absent."""
    if not os.path.exists(npz_path):
        return None, None, 0
    z = np.load(npz_path)
    per, pooled = [], []
    for k in z.files:
        arr = np.array(z[k])                      # (N_rows, n_taus)
        if arr.size == 0:
            continue
        pooled.append(arr)
        per.append(np.mean([[(arr[:, i] < th).mean() for th in VSD_THS]
                            for i in range(arr.shape[1])]))
    allr = np.concatenate(pooled, axis=0)
    flat = np.mean([[(allr[:, i] < th).mean() for th in VSD_THS]
                    for i in range(allr.shape[1])])
    return float(flat), float(np.mean(per)), len(allr)


def main():
    csv_path, bop_root = sys.argv[1], Path(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else Path(csv_path).name
    e = errors(csv_path, bop_root)
    f_mssd, f_mspd, p_mssd, p_mspd, n = agg(e)
    f_vsd, p_vsd, nv = vsd_from_npz(str(csv_path) + ".vsd_errs.npz")
    print(f"=== {label}  ({n} scored rows"
          + (f", {nv} VSD rows" if nv else ", no VSD npz") + ") ===")
    print(f"{'':16}{'MSSD':>9}{'MSPD':>9}{'VSD':>9}{'full AR':>10}{'AR(2/3)':>10}")
    for name, a, b, c in (("BOP flat", f_mssd, f_mspd, f_vsd),
                          ("popoe per-obj", p_mssd, p_mspd, p_vsd)):
        full = f"{(a + b + c) / 3:.4f}" if c is not None else "n/a"
        print(f"{name:16}{a:9.4f}{b:9.4f}"
              + (f"{c:9.4f}" if c is not None else f"{'n/a':>9}")
              + f"{full:>10}{(a + b) / 2:10.4f}")


if __name__ == "__main__":
    main()
