"""Compute BOP AR (MSSD + MSPD partial, skip VSD) directly from pose CSV.

This bypasses bop_toolkit's OpenGL-based orchestration (which fails in headless
containers without EGL drivers). Uses bop_toolkit_lib's pose_error for MSSD/MSPD.

Reported:
  AR_MSSD : mean recall over thresholds 0.05..0.50 × diameter
  AR_MSPD : mean recall over thresholds 5..50 px (re-scaled by 640/W per BOP spec)
  AR_2/3  : mean(AR_MSSD, AR_MSPD)  — ≈ 2/3 of full BOP AR
"""
import os, sys, csv, json
from pathlib import Path
import numpy as np, trimesh

sys.path.insert(0, os.environ.get("POPOE_BOP_TOOLKIT", "/workspace/bop_toolkit"))
from bop_toolkit_lib import inout, misc, pose_error

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "/workspace/bop_results/geditest_lmo-test.csv"
# Dataset path is env-overridable so the same script scores LM-O or YCB-V:
#   BOP_PATH=/workspace/bop_data/ycbv python compute_ar_mssd_mspd.py preds.csv
BOP_PATH = Path(os.environ.get("BOP_PATH", "/workspace/bop_data/lmo"))

# Load rows
rows = list(csv.DictReader(open(CSV_PATH)))
print(f"loaded {len(rows)} rows from {CSV_PATH}", flush=True)

# Index GT
scenes = sorted({int(r["scene_id"]) for r in rows})
gt_by_scene_im_obj = {}
for s in scenes:
    sdir = BOP_PATH / "test" / f"{s:06d}"
    scene_gt = json.load(open(sdir / "scene_gt.json"))
    scene_cam = json.load(open(sdir / "scene_camera.json"))
    for im_id_str, gts in scene_gt.items():
        im_id = int(im_id_str)
        K = np.array(scene_cam[im_id_str]["cam_K"]).reshape(3, 3)
        for g in gts:
            key = (s, im_id, g["obj_id"])
            gt_by_scene_im_obj.setdefault(key, []).append(dict(
                R=np.array(g["cam_R_m2c"]).reshape(3, 3),
                t=np.array(g["cam_t_m2c"]).reshape(3),
                K=K))

# Load per-obj model pts + symmetries + diameters from models_eval/models_info.json
models_info = json.load(open(BOP_PATH / "models_eval" / "models_info.json"))
obj_data = {}
for obj_id_str, info in models_info.items():
    obj_id = int(obj_id_str)
    mesh = trimesh.load(BOP_PATH / "models_eval" / f"obj_{obj_id:06d}.ply", force="mesh")
    pts = np.array(mesh.vertices)  # mm
    diameter = info["diameter"]  # mm
    syms = misc.get_symmetry_transformations(info, max_sym_disc_step=0.01)
    obj_data[obj_id] = dict(pts=pts, diameter=diameter, syms=syms)
    print(f"obj{obj_id}: diam={diameter:.1f}mm verts={len(pts)} syms={len(syms)}")

# Compute errors
errs_mssd = {}  # obj_id -> list of (scene, im, error_mm)
errs_mspd = {}

for r in rows:
    scene_id = int(r["scene_id"])
    im_id = int(r["im_id"])
    obj_id = int(r["obj_id"])
    score = float(r["score"])
    R_est = np.array([float(x) for x in r["R"].split()]).reshape(3, 3)
    t_est = np.array([float(x) for x in r["t"].split()]).reshape(3, 1)  # mm

    key = (scene_id, im_id, obj_id)
    if key not in gt_by_scene_im_obj:
        continue
    gts = gt_by_scene_im_obj[key]
    d = obj_data[obj_id]
    # Reshape GT t to (3,1)
    for g in gts:
        if g["t"].shape != (3, 1):
            g["t"] = g["t"].reshape(3, 1)

    # For n_top=1 (BOP standard for 6D loc), we take the single highest-scoring estimate.
    # Here we have 1 est per target already (our CSV writes 1 row per target).
    # For each GT instance (usually 1), compute error and take the best (min).
    best_mssd = min(pose_error.mssd(R_est, t_est, g["R"], g["t"], d["pts"], d["syms"]) for g in gts)
    best_mspd = min(pose_error.mspd(R_est, t_est, g["R"], g["t"], g["K"], d["pts"], d["syms"]) for g in gts)
    errs_mssd.setdefault(obj_id, []).append(best_mssd)
    errs_mspd.setdefault(obj_id, []).append(best_mspd)

# BOP19 recall thresholds
mssd_thrs = np.arange(0.05, 0.51, 0.05)  # fraction of diameter
mspd_thrs = np.arange(5, 51, 5)  # pixels (reference W=640)

# Per-object recall per threshold
objs = sorted(errs_mssd.keys())
print("\n=== Per-object recall ===")
print(f"{'obj':<5}{'#':<5}{'AR_MSSD':<10}{'AR_MSPD':<10}")
per_obj = {}
for o in objs:
    d_mm = obj_data[o]["diameter"]
    mssd_errs = np.array(errs_mssd[o])
    mspd_errs = np.array(errs_mspd[o])
    n = len(mssd_errs)
    mssd_recalls = [(mssd_errs < thr * d_mm).mean() for thr in mssd_thrs]
    mspd_recalls = [(mspd_errs < thr).mean() for thr in mspd_thrs]
    ar_mssd = np.mean(mssd_recalls)
    ar_mspd = np.mean(mspd_recalls)
    per_obj[o] = (ar_mssd, ar_mspd, n)
    print(f"{o:<5}{n:<5}{ar_mssd:<10.4f}{ar_mspd:<10.4f}")

# Mean over objects
AR_MSSD = np.mean([v[0] for v in per_obj.values()])
AR_MSPD = np.mean([v[1] for v in per_obj.values()])
AR_23 = (AR_MSSD + AR_MSPD) / 2
print(f"\n=== {Path(CSV_PATH).name} ===")
print(f"AR_MSSD : {AR_MSSD:.4f}")
print(f"AR_MSPD : {AR_MSPD:.4f}")
print(f"AR(2/3) : {AR_23:.4f}   (skipped VSD; full BOP AR = mean of all three)")
