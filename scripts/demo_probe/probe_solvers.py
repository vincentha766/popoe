"""Compare registration back-ends on real query/target features: O3D (CPU) vs GPU RANSAC."""
import os, sys, time, json, glob
import numpy as np
sys.path.insert(0, "/workspace/popoe_vramprobe/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_common import mark, dump, smi_total

import torch, cv2
SCENE, IMID, CACHE = sys.argv[1], sys.argv[2], sys.argv[3]

torch.zeros(1, device="cuda")
from popoe.freeze.feature_extractor import TargetFeatureExtractor, load_dinov2, load_geometric_descriptor
from popoe.interfaces import PointFeatures, CanonFrame
from popoe.solvers.open3d_ransac import Open3DFeatureRansacSolver
from popoe.solvers.gpu_ransac import GPURansacSolver

tfe = TargetFeatureExtractor(device="cuda")
mark("00 pose stack resident (DINOv2-g + GeDi x2)")
tfe._canon_scale = 1.0

rgb = cv2.cvtColor(cv2.imread(f"{SCENE}/rgb/{IMID}.png"), cv2.COLOR_BGR2RGB)
cam = json.load(open(f"{SCENE}/scene_camera.json"))[str(int(IMID))]
K = np.array(cam["cam_K"]).reshape(3, 3)
depth = cv2.imread(f"{SCENE}/depth/{IMID}.png", cv2.IMREAD_UNCHANGED).astype(np.float32) \
        * cam["depth_scale"] / 1000.0
intr = dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
mpath = sorted(glob.glob(f"{SCENE}/mask_visib/{IMID}_*.png"))[0]
m = cv2.imread(mpath, cv2.IMREAD_UNCHANGED) > 0
pts, feats = tfe.extract_target_features(rgb, depth, m, intr)

ys, xs = np.where((depth > 0) & m)
dd = depth[ys, xs]
dense = np.stack([(xs - intr['cx']) * dd / intr['fx'],
                  (ys - intr['cy']) * dd / intr['fy'], dd], 1).astype(np.float32)

q = np.load(sorted(glob.glob(f"{CACHE}/query_*.npz"))[0])
query = PointFeatures(pts=q["pts"].astype(np.float64), feats=q["feats"].astype(np.float64))
target = PointFeatures(pts=pts.astype(np.float64), feats=feats.astype(np.float64),
                       pts_dense=dense.astype(np.float64))
frame = CanonFrame(center=np.zeros(3), scale=1.0)
print(f"query {query.pts.shape} target {target.pts.shape}", flush=True)

def bench(name, solver, n=3):
    solver.solve(query, target, frame)          # warmup
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.time()
        hyps = solver.solve(query, target, frame)
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    print(f"[SOLVE] {name:34s} mean {np.mean(ts):6.3f} s  "
          f"(min {min(ts):.3f} max {max(ts):.3f})  hyps={len(hyps)}", flush=True)
    return np.mean(ts)

t_o3d = bench("Open3D RANSAC (CPU, headline)", Open3DFeatureRansacSolver())
t_gpu = bench("GPU RANSAC 10k iters", GPURansacSolver(iters=10000))
mark("01 after GPU RANSAC (peak)")

u, tot = smi_total()
print(f"\n[GPU] whole card used {u} / {tot} MiB")
print(f"[SUM] O3D CPU {t_o3d:.3f}s vs GPU {t_gpu:.3f}s per hypothesis set "
      f"-> speedup {t_o3d/max(t_gpu,1e-6):.0f}x")
dump("/workspace/probe_out/solver_marks.json")
