"""Compute BOP AR (MSSD + MSPD partial, skip VSD) directly from pose CSV.

This bypasses bop_toolkit's OpenGL-based orchestration (which fails in headless
containers without EGL drivers). Uses bop_toolkit_lib's pose_error for MSSD/MSPD.

Reported (BOP official FLAT calibre — every annotated instance carries equal
weight; see metrics/aggregate.py):
  AR_MSSD : mean recall over thresholds 0.05..0.50 × diameter
  AR_MSPD : mean recall over thresholds 5..50 px (re-scaled by 640/W per BOP spec)
  AR_2/3  : mean(AR_MSSD, AR_MSPD)  — ≈ 2/3 of full BOP AR
Per-object equal-weight means (this script's former, under-reporting calibre)
are still printed as clearly-labelled secondary output for reconciling against
historical records.
"""
import os, sys, csv, json
from pathlib import Path
import numpy as np, trimesh

try:
    from popoe.metrics import aggregate
except ImportError:  # run as a bare file without popoe installed
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from popoe.metrics import aggregate

sys.path.insert(0, os.environ.get("POPOE_BOP_TOOLKIT", "/workspace/bop_toolkit"))
from bop_toolkit_lib import inout, misc, pose_error

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "/workspace/bop_results/geditest_lmo-test.csv"
# Dataset path is env-overridable so the same script scores LM-O or YCB-V:
#   BOP_PATH=/workspace/bop_data/ycbv python compute_ar_mssd_mspd.py preds.csv
BOP_PATH = Path(os.environ.get("BOP_PATH", "/workspace/bop_data/lmo"))

# Load rows
rows = list(csv.DictReader(open(CSV_PATH)))
print(f"loaded {len(rows)} rows from {CSV_PATH}", flush=True)

# This scorer matches each row independently to its BEST GT instance — sound
# only when there is ONE row per (scene, im, obj). With multi-instance rows
# (inst_count > 1) two estimates could both claim the same GT while another
# instance goes unpenalised, silently inflating AR. Fail loudly instead.
_keys = [(r["scene_id"], r["im_id"], r["obj_id"]) for r in rows]
if len(_keys) != len(set(_keys)):
    raise SystemExit(
        "multi-instance CSV detected (duplicate (scene,im,obj) rows): this "
        "local AR scorer assumes one row per target. Score with the official "
        "bop_toolkit, or extend this script with one-to-one GT assignment.")

# M1: flat AR denominators by CSV alone inflate if targets are missing.
_targets = BOP_PATH / "test_targets_bop19.json"
aggregate.assert_csv_covers_targets(_keys, _targets)

# Default image widths (BOP dataset_params) when no depth/rgb is on disk.
_DATASET_IM_WIDTH = {
    "lmo": 640, "lm": 640, "ycbv": 640, "tless": 720, "itodd": 1280,
    "hb": 640, "icbin": 640, "tudl": 640, "hope": 640, "ruapc": 640,
}
_ds_name = BOP_PATH.name.lower()
_default_w = float(os.environ.get(
    "BOP_IMAGE_WIDTH", _DATASET_IM_WIDTH.get(_ds_name, aggregate.MSPD_REF_WIDTH)))

# Index GT + per-scene image width (for MSPD 640/W normalisation).
scenes = sorted({int(r["scene_id"]) for r in rows})
gt_by_scene_im_obj = {}
scene_im_width = {}  # scene_id -> W
for s in scenes:
    sdir = BOP_PATH / "test" / f"{s:06d}"
    scene_gt = json.load(open(sdir / "scene_gt.json"))
    scene_cam = json.load(open(sdir / "scene_camera.json"))
    # Prefer real image size (depth/rgb) so T-LESS/ITODD get the right W.
    w = None
    for im_id_str in scene_gt.keys():
        for sub in ("depth", "rgb", "gray"):
            p = sdir / sub / f"{int(im_id_str):06d}.png"
            if p.is_file():
                try:
                    import cv2
                    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    if im is not None:
                        w = float(im.shape[1])
                        break
                except Exception:
                    pass
        if w is not None:
            break
    scene_im_width[s] = w if w is not None else _default_w
    for im_id_str, gts in scene_gt.items():
        im_id = int(im_id_str)
        K = np.array(scene_cam[im_id_str]["cam_K"]).reshape(3, 3)
        for g in gts:
            key = (s, im_id, g["obj_id"])
            gt_by_scene_im_obj.setdefault(key, []).append(dict(
                R=np.array(g["cam_R_m2c"]).reshape(3, 3),
                t=np.array(g["cam_t_m2c"]).reshape(3),
                K=K))
print(f"MSPD image widths (scenes): "
      f"min={min(scene_im_width.values()):.0f} "
      f"max={max(scene_im_width.values()):.0f} "
      f"(dataset default {_default_w:.0f})", flush=True)
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
    # N3: BOP normalises MSPD by image width before thresholding
    # (eval_calc_scores.py: err *= 640/W). Store the *normalised* error so
    # fixed thr in {5..50} stay correct for T-LESS (720) / ITODD (1280).
    factor = aggregate.mspd_normalize_factor(scene_im_width[scene_id])
    best_mspd_norm = best_mspd * factor
    errs_mssd.setdefault(obj_id, []).append(best_mssd)
    errs_mspd.setdefault(obj_id, []).append(best_mspd_norm)

# BOP19 recall thresholds
mssd_thrs = aggregate.MSSD_THRS  # fraction of diameter
mspd_thrs = aggregate.MSPD_THRS  # px on the 640-wide reference frame

# Per-object recall per threshold (diagnostic table; also feeds the legacy
# per-object means below)
objs = sorted(errs_mssd.keys())
print("\n=== Per-object recall ===")
print(f"{'obj':<5}{'#':<5}{'AR_MSSD':<10}{'AR_MSPD':<10}")
per_obj = {}
for o in objs:
    d_mm = obj_data[o]["diameter"]
    mssd_errs = np.array(errs_mssd[o])
    mspd_errs = np.array(errs_mspd[o])  # already 640-normalised
    n = len(mssd_errs)
    mssd_recalls = [(mssd_errs < thr * d_mm).mean() for thr in mssd_thrs]
    mspd_recalls = [(mspd_errs < thr).mean() for thr in mspd_thrs]
    ar_mssd = np.mean(mssd_recalls)
    ar_mspd = np.mean(mspd_recalls)
    per_obj[o] = (ar_mssd, ar_mspd, n)
    print(f"{o:<5}{n:<5}{ar_mssd:<10.4f}{ar_mspd:<10.4f}")

# PRIMARY: flat aggregation over all annotated instances (BOP official).
# The former equal-weight-per-object mean under-reported 0.5-0.9 pt on
# LM-O/YCB-V; flat matches the BOP server to 0.03 pt (see metrics/aggregate.py).
# M3 call-site: headline MUST use flat_ar_*; tests assert this symbol path.
all_mssd = np.concatenate([np.asarray(errs_mssd[o], dtype=float) for o in objs])
all_mspd = np.concatenate([np.asarray(errs_mspd[o], dtype=float) for o in objs])
all_diam = np.concatenate([np.full(len(errs_mssd[o]), obj_data[o]["diameter"])
                           for o in objs])
AR_MSSD = aggregate.flat_ar_mssd(all_mssd, all_diam, mssd_thrs)
AR_MSPD = aggregate.flat_ar_mspd(all_mspd, mspd_thrs)
AR_23 = (AR_MSSD + AR_MSPD) / 2

# SECONDARY: legacy per-object equal-weight means, only for reconciling
# against records produced before the calibre fix. Do not quote as BOP AR.
AR_MSSD_PEROBJ = float(np.mean([v[0] for v in per_obj.values()]))
AR_MSPD_PEROBJ = float(np.mean([v[1] for v in per_obj.values()]))

print(f"\n=== {Path(CSV_PATH).name} ===")
print(f"AR_MSSD : {AR_MSSD:.4f}")
print(f"AR_MSPD : {AR_MSPD:.4f}")
print(f"AR(2/3) : {AR_23:.4f}   (skipped VSD; full BOP AR = mean of all three)")
print(f"[legacy per-object calibre] AR_MSSD {AR_MSSD_PEROBJ:.4f}  "
      f"AR_MSPD {AR_MSPD_PEROBJ:.4f}  AR(2/3) "
      f"{(AR_MSSD_PEROBJ + AR_MSPD_PEROBJ) / 2:.4f}")
