"""popoe pose route: VRAM + latency for DINOv2-g + two-scale GeDi + O3D RANSAC/ICP."""
import os, sys, time, json, glob, pickle
import numpy as np
sys.path.insert(0, "/workspace/popoe_vramprobe/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_common import mark, dump, smi_total

import torch, cv2

SCENE = sys.argv[1]          # e.g. /workspace/bop_data/ycbv/test/000048
IMID = sys.argv[2]           # e.g. 000001
CACHE = sys.argv[3]          # e.g. /workspace/popoe_cache_ycbv_v5

torch.zeros(1, device="cuda")
mark("00 cuda context only")

from popoe.freeze.feature_extractor import (TargetFeatureExtractor,
                                            load_dinov2, load_gedi)

dino = load_dinov2("cuda")
mark("01 +DINOv2-g")
gedi = load_gedi("cuda")
mark("02 +two-scale GeDi (BOTH RESIDENT)")

tfe = TargetFeatureExtractor(device="cuda", dino=dino, gedi=gedi)

# ── load a real YCB-V frame: rgb + depth(m) + per-instance visible masks ──
rgb = cv2.cvtColor(cv2.imread(f"{SCENE}/rgb/{IMID}.png"), cv2.COLOR_BGR2RGB)
scene_cam = json.load(open(f"{SCENE}/scene_camera.json"))[str(int(IMID))]
K = np.array(scene_cam["cam_K"]).reshape(3, 3)
dscale = scene_cam["depth_scale"]
depth = cv2.imread(f"{SCENE}/depth/{IMID}.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
depth = depth * dscale / 1000.0            # -> metres
intr = dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])

masks = sorted(glob.glob(f"{SCENE}/mask_visib/{IMID}_*.png"))
print(f"frame {SCENE}/{IMID}: rgb {rgb.shape}, {len(masks)} GT instance masks", flush=True)

tfe._canon_scale = 1.0

def encode(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) > 0
    torch.cuda.synchronize(); t0 = time.time()
    pts, feats = tfe.extract_target_features(rgb, depth, m, intr)
    torch.cuda.synchronize()
    return time.time() - t0, pts, feats

print("\n--- warmup ---", flush=True)
dt, pts, feats = encode(masks[0])
print(f"[WARM] encode 1 mask {dt:.2f}s  pts={None if pts is None else pts.shape}", flush=True)
mark("03 after warmup encode")

print("\n--- steady state: per-detection target encode ---", flush=True)
ts = []
for mp in masks[:8]:
    dt, pts, feats = encode(mp)
    ts.append(dt)
    print(f"[ENC] {os.path.basename(mp)}  {dt:5.2f}s  "
          f"pts={None if pts is None else pts.shape[0]:4d}", flush=True)
mark("04 after 8 encodes (peak)")

# ── registration: match + feature-aware RANSAC + ICP, on real query features ──
from popoe.registration import (top_k_correspondences, ransac_pose_estimation,
                                icp_refinement)
qs = sorted(glob.glob(f"{CACHE}/query_*.npz"))
print(f"\n{len(qs)} cached query feature files; using {os.path.basename(qs[0])}", flush=True)
q = np.load(qs[0])
print("   query npz keys:", list(q.keys()), flush=True)
qpts = q[[k for k in q.keys() if "pt" in k.lower()][0]]
qfeat = q[[k for k in q.keys() if "feat" in k.lower()][0]]
print(f"   query pts {qpts.shape} feats {qfeat.shape}; target pts {pts.shape} "
      f"feats {feats.shape}", flush=True)

t0 = time.time()
R, t, sc = ransac_pose_estimation(qpts, qfeat, pts, feats,
                                  n_iters=10000, tau_inlier=0.02)
t_ransac = time.time() - t0
print(f"[TIME] feature-aware RANSAC (10k iters, CPU/numpy)  {t_ransac:7.2f} s", flush=True)

ys, xs = np.where((depth > 0) & (cv2.imread(masks[0], cv2.IMREAD_UNCHANGED) > 0))
dd = depth[ys, xs]
dense = np.stack([(xs - intr['cx']) * dd / intr['fx'],
                  (ys - intr['cy']) * dd / intr['fy'], dd], 1).astype(np.float32)
t0 = time.time()
icp_refinement(qpts, dense, R, t)
t_icp = time.time() - t0
print(f"[TIME] ICP refinement (Open3D, CPU)                 {t_icp:7.2f} s", flush=True)

u, tot = smi_total()
print(f"\n[GPU] whole card used {u} / {tot} MiB with the pose route resident")
print(f"[SUM] target encode: mean {np.mean(ts):.2f} s/detection "
      f"(min {min(ts):.2f}, max {max(ts):.2f}, n={len(ts)})")
print(f"[SUM] registration per hypothesis: RANSAC {t_ransac:.2f}s + ICP {t_icp:.2f}s (CPU)")
dump("/workspace/probe_out/pose_marks.json")
