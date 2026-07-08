"""VSD (Visible Surface Discrepancy) via nvdiffrast, for BOP AR eval.

Adds the missing 3rd metric to our AR computation. Uses our existing
nvdiffrast infrastructure (no OpenGL/EGL needed).

BOP19 spec:
  - Render object depth at est_pose and gt_pose
  - Visibility mask: rendered depth within delta (15mm) of scene depth
  - VSD_tau = 1 - |vis_inter AND |d_est-d_gt|<tau*diam| / |vis_union|
  - Thresholds tau in [0.05, 0.50] step 0.05 (normalized by diameter)
  - Recall at each tau, average → AR_VSD
"""
import os, sys, json, csv
from pathlib import Path
import numpy as np, cv2, torch, trimesh
import nvdiffrast.torch as dr

sys.path.insert(0, os.environ.get("POPOE_BOP_TOOLKIT", "/workspace/bop_toolkit"))
from bop_toolkit_lib import misc

VSD_DELTA_MM = 15.0  # BOP standard
VSD_TAUS = np.arange(0.05, 0.51, 0.05)


_ctx = None
def _get_ctx(device="cuda"):
    global _ctx
    if _ctx is None:
        _ctx = dr.RasterizeCudaContext(device=device)
    return _ctx


@torch.no_grad()
def render_depth_k(V_np, F_np, R, t, K, H, W, device="cuda"):
    """Render object depth map using camera intrinsics K (mm units).

    V_np: (V, 3) vertices in object frame (mm)
    F_np: (F, 3) int32
    R: (3, 3), t: (3,) pose in mm (object -> camera)
    K: (3, 3) camera intrinsics
    Returns: (H, W) float32 depth in mm, 0 where no hit.
    """
    ctx = _get_ctx(device)
    V_np = V_np.astype(np.float32)
    F_np = F_np.astype(np.int32)
    R = R.astype(np.float32)
    t = t.astype(np.float32).reshape(3)
    K = K.astype(np.float32)

    # Transform to camera frame
    V_cam = V_np @ R.T + t  # (V, 3) in mm

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # OpenCV intrinsics → nvdiffrast clip space (same convention as FE):
    # col = x_cam/z*fx + cx, row = y_cam/z*fy + cy (y down in cam frame)
    # NDC: x_ndc = (2*col/W) - 1, y_ndc = (2*row/H) - 1  (y_ndc+1 = bottom per our earlier test)
    x_clip = (2 * fx / W) * V_cam[:, 0] + (2 * cx / W - 1) * V_cam[:, 2]
    y_clip = (2 * fy / H) * V_cam[:, 1] + (2 * cy / H - 1) * V_cam[:, 2]
    w_clip = V_cam[:, 2]
    # z_clip: standard OpenGL-like, with near/far chosen large enough for mm-scale mesh
    near, far = 1.0, 1.0e6
    z_clip = ((far + near) / (far - near)) * V_cam[:, 2] - (2 * far * near) / (far - near)
    V_clip = np.stack([x_clip, y_clip, z_clip, w_clip], axis=1).astype(np.float32)

    pos = torch.from_numpy(V_clip).unsqueeze(0).to(device).contiguous()
    tri = torch.from_numpy(F_np).to(device).contiguous()
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=[H, W])  # (1, H, W, 4)
    hit = rast[0, :, :, 3] > 0

    # Interpolate z_cam per pixel
    zcam_t = torch.from_numpy(V_cam[:, 2:3].astype(np.float32)).unsqueeze(0).to(device)
    z_interp, _ = dr.interpolate(zcam_t, rast, tri)  # (1, H, W, 1)
    depth = z_interp[0, :, :, 0]
    depth = torch.where(hit, depth, torch.zeros_like(depth)).cpu().numpy().astype(np.float32)
    return depth


def vsd_per_tau(d_est, d_gt, d_scene, diameter_mm, delta_mm=VSD_DELTA_MM, taus=VSD_TAUS):
    """BOP19 VSD per tau. All depths in mm."""
    # Visibility:
    #   est visible if d_est > 0 AND (d_scene <= 0 OR d_est <= d_scene + delta)
    #   gt similarly
    scene_missing = d_scene <= 0
    vis_est = (d_est > 0) & (scene_missing | (d_est <= d_scene + delta_mm))
    vis_gt = (d_gt > 0) & (scene_missing | (d_gt <= d_scene + delta_mm))
    vis_inter = vis_est & vis_gt
    vis_union = vis_est | vis_gt

    dist = np.abs(d_est.astype(np.float64) - d_gt.astype(np.float64))
    if vis_union.sum() == 0:
        return [1.0] * len(taus)
    errs = []
    for tau in taus:
        thr = tau * diameter_mm
        correct = dist < thr
        err = 1.0 - (vis_inter & correct).sum() / vis_union.sum()
        errs.append(float(err))
    return errs


def compute_ar_vsd(csv_path, bop_root, models_eval_dir=None):
    """Read BOP CSV + scene depth + GT, compute AR_VSD.

    models_eval_dir: path to the dataset's models_eval folder (uses these meshes
    for VSD, per BOP protocol)
    """
    bop_root = Path(bop_root)
    if models_eval_dir is None:
        models_eval_dir = bop_root / "models_eval"
    models_eval_dir = Path(models_eval_dir)

    # Load meshes + symmetries + diameters once
    models_info = json.load(open(models_eval_dir / "models_info.json"))
    obj_data = {}
    for obj_id_str, info in models_info.items():
        obj_id = int(obj_id_str)
        m = trimesh.load(models_eval_dir / f"obj_{obj_id:06d}.ply", force="mesh")
        syms = misc.get_symmetry_transformations(info, max_sym_disc_step=0.01)
        # Cap for VSD speed (render per sym; YCB-V obj18 has 315 → 18h full eval)
        if len(syms) > 32:
            step = max(1, len(syms) // 32)
            syms = syms[::step][:32]
        obj_data[obj_id] = dict(
            V=np.array(m.vertices, dtype=np.float32),
            F=np.array(m.faces, dtype=np.int32),
            diameter=info["diameter"],
            syms=syms,
        )
        print(f"obj{obj_id}: diam={info['diameter']:.1f}mm V={len(m.vertices)} syms={len(obj_data[obj_id]['syms'])}", flush=True)

    rows = list(csv.DictReader(open(csv_path)))
    print(f"loaded {len(rows)} rows", flush=True)

    # Index GT + scene depth
    scene_cache = {}

    errs_per_obj = {}

    for i, r in enumerate(rows):
        scene_id = int(r["scene_id"])
        im_id = int(r["im_id"])
        obj_id = int(r["obj_id"])
        R_est = np.array([float(x) for x in r["R"].split()]).reshape(3, 3)
        t_est = np.array([float(x) for x in r["t"].split()]).reshape(3)  # mm

        key = (scene_id, im_id)
        if key not in scene_cache:
            scene_dir = bop_root / "test" / f"{scene_id:06d}"
            scene_camera = json.load(open(scene_dir / "scene_camera.json"))
            scene_gt = json.load(open(scene_dir / "scene_gt.json"))
            cam = scene_camera[str(im_id)]
            K = np.array(cam["cam_K"]).reshape(3, 3)
            depth_raw = cv2.imread(str(scene_dir / "depth" / f"{im_id:06d}.png"), cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                scene_cache[key] = None
                continue
            depth_scene = depth_raw.astype(np.float32) * cam["depth_scale"]  # to mm
            scene_cache[key] = dict(K=K, depth=depth_scene, gt=scene_gt[str(im_id)],
                                     H=depth_raw.shape[0], W=depth_raw.shape[1])
        sc = scene_cache[key]
        if sc is None:
            errs_per_obj.setdefault(obj_id, []).append([1.0] * len(VSD_TAUS))
            continue
        gt_matches = [g for g in sc["gt"] if g["obj_id"] == obj_id]
        if not gt_matches:
            errs_per_obj.setdefault(obj_id, []).append([1.0] * len(VSD_TAUS))
            continue

        d = obj_data[obj_id]
        # Render depth at estimated pose
        d_est = render_depth_k(d["V"], d["F"], R_est, t_est, sc["K"], sc["H"], sc["W"])

        # Best over GT instances × symmetries (BOP protocol)
        best_err = None
        for g in gt_matches:
            R_gt = np.array(g["cam_R_m2c"]).reshape(3, 3)
            t_gt = np.array(g["cam_t_m2c"]).reshape(3)
            for sym in d["syms"]:
                R_gt_s = R_gt @ sym["R"]
                t_gt_s = (R_gt @ sym["t"].reshape(3, 1)).reshape(3) + t_gt
                d_gt = render_depth_k(d["V"], d["F"], R_gt_s, t_gt_s, sc["K"], sc["H"], sc["W"])
                errs = vsd_per_tau(d_est, d_gt, sc["depth"], d["diameter"])
                if best_err is None or sum(errs) < sum(best_err):
                    best_err = errs
        errs_per_obj.setdefault(obj_id, []).append(best_err)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(rows)}]", flush=True)

    # Persist raw per-row per-tau errors so aggregation-protocol changes never
    # require re-rendering.
    raw_out = str(csv_path) + ".vsd_errs.npz"
    np.savez(raw_out, **{f"obj{oid}": np.array(v) for oid, v in errs_per_obj.items()})
    print(f"raw errors -> {raw_out}")

    # Aggregate per-obj AR, then mean.
    # BOP19 protocol: recall over the FULL tau x threshold grid
    # (tau in 0.05..0.5 AND correctness threshold th in 0.05..0.5, 10x10 cells)
    # — NOT a fixed th=0.3 (the old bug here, systematically biased AR_VSD).
    VSD_THS = np.arange(0.05, 0.51, 0.05)
    print("\n=== Per-object AR_VSD (BOP19 tau x th grid) ===")
    print(f"{'obj':<5}{'#':<5}{'AR_VSD':<10}")
    per_obj_ar = {}
    for obj_id in sorted(errs_per_obj.keys()):
        arr = np.array(errs_per_obj[obj_id])  # (N, n_taus)
        rec = np.array([[(arr[:, i] < th).mean() for th in VSD_THS]
                        for i in range(arr.shape[1])])
        ar = float(np.mean(rec))
        per_obj_ar[obj_id] = (ar, len(arr))
        print(f"{obj_id:<5}{len(arr):<5}{ar:<10.4f}")
    AR_VSD = np.mean([v[0] for v in per_obj_ar.values()])
    print(f"\nAR_VSD : {AR_VSD:.4f}")
    return AR_VSD


if __name__ == "__main__":
    csv_path = sys.argv[1]
    bop_root = sys.argv[2]
    compute_ar_vsd(csv_path, bop_root)
