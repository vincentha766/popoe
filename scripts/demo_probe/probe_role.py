"""Hold one pipeline role resident and report its VRAM; used for the concurrency test.

usage: probe_role.py <role> <rgb_png> <hold_seconds>
roles: muse | cnos_amg | pose | sam6d_ism
"""
import os, sys, time, json, glob
import numpy as np
sys.path.insert(0, "/workspace/popoe_vramprobe/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_common import smi_total

import torch, cv2
ROLE, RGB, HOLD = sys.argv[1], sys.argv[2], float(sys.argv[3])
# Optional hard cap (MiB) so the process behaves as it would on a smaller card.
CAP = float(sys.argv[4]) if len(sys.argv) > 4 else 0
rgb = cv2.cvtColor(cv2.imread(RGB), cv2.COLOR_BGR2RGB)

torch.zeros(1, device="cuda")
if CAP:
    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
    torch.cuda.set_per_process_memory_fraction(min(1.0, CAP / total_mib))
    print(f"[{ROLE}] CAP {CAP:.0f} MiB "
          f"(fraction {CAP/total_mib:.3f} of {total_mib:.0f})", flush=True)
base = smi_total()[0]
t0 = time.time()

def report(stage, extra=""):
    u, tot = smi_total()
    print(f"[{ROLE}] {stage:26s} card={u:6d} MiB | torch alloc="
          f"{torch.cuda.memory_allocated()//2**20:5d} peak_alloc="
          f"{torch.cuda.max_memory_allocated()//2**20:5d} reserved="
          f"{torch.cuda.memory_reserved()//2**20:5d} MiB | "
          f"t={time.time()-t0:6.1f}s {extra}", flush=True)

report("cuda context")

if ROLE == "muse":
    from dataclasses import dataclass
    from popoe.segmentor_muse import (GroundingDinoBoxProposer, SAM2BoxMaskRefiner,
                                      DinoV2ClsGemEmbedder)
    from popoe.segmentor_cnos_v3 import square_crop
    @dataclass
    class S: rgb: np.ndarray; depth: np.ndarray = None
    p = GroundingDinoBoxProposer(device="cuda"); p._load()
    r = SAM2BoxMaskRefiner(device="cuda", model_size="large",
                           sam_ckpt_dir="/workspace/sam2_ckpt"); r._load()
    e = DinoV2ClsGemEmbedder(device="cuda"); _ = e.model
    report("models loaded")
    def work():
        b, _ = p.propose(S(rgb=rgb))
        for m, _i in r.masks_for_boxes(rgb, b):
            c, cm = square_crop(rgb, m)
            if c is not None: e.embed(c, cm)
        return len(b)

elif ROLE == "cnos_amg":
    from popoe.segmentor_cnos_v3 import DinoV2ForegroundPatchExtractor as DinoV2PatchExtractor
    from popoe.segmentor import build_sam2_model, AMG_PARAMS
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    sam = build_sam2_model("large", "cuda", "/workspace/sam2_ckpt")
    gen = SAM2AutomaticMaskGenerator(sam, **AMG_PARAMS)
    d = DinoV2PatchExtractor(device="cuda"); _ = d.model if hasattr(d, "model") else None
    report("models loaded")
    def work():
        return len(gen.generate(rgb))

elif ROLE == "pose":
    from popoe.freeze.feature_extractor import TargetFeatureExtractor
    from popoe.interfaces import PointFeatures, CanonFrame
    from popoe.solvers.open3d_ransac import Open3DFeatureRansacSolver
    tfe = TargetFeatureExtractor(device="cuda"); tfe._canon_scale = 1.0
    solver = Open3DFeatureRansacSolver()
    report("models loaded")
    SCENE = os.path.dirname(os.path.dirname(RGB)); IMID = os.path.basename(RGB)[:-4]
    cam = json.load(open(f"{SCENE}/scene_camera.json"))[str(int(IMID))]
    K = np.array(cam["cam_K"]).reshape(3, 3)
    depth = cv2.imread(f"{SCENE}/depth/{IMID}.png", cv2.IMREAD_UNCHANGED).astype(np.float32) \
            * cam["depth_scale"] / 1000.0
    intr = dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
    masks = sorted(glob.glob(f"{SCENE}/mask_visib/{IMID}_*.png"))
    q = np.load(sorted(glob.glob("/workspace/popoe_cache_ycbv_v5/query_*.npz"))[0])
    query = PointFeatures(pts=q["pts"].astype(np.float64), feats=q["feats"].astype(np.float64))
    frame = CanonFrame(center=np.zeros(3), scale=1.0)
    def work():
        n = 0
        for mp in masks:
            m = cv2.imread(mp, cv2.IMREAD_UNCHANGED) > 0
            pts, feats = tfe.extract_target_features(rgb, depth, m, intr)
            if pts is None: continue
            solver.solve(query, PointFeatures(pts=pts.astype(np.float64),
                                              feats=feats.astype(np.float64)), frame)
            n += 1
        return n

elif ROLE == "sam6d_ism":
    # The composition actually on this volume: SAM ViT-H proposals + DINOv2 ViT-L
    # descriptors (the FastSAM checkpoint dir is empty).
    ISM = "/workspace/SAM-6D/SAM-6D/Instance_Segmentation_Model/checkpoints"
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_h"](checkpoint=f"{ISM}/segment-anything/sam_vit_h_4b8939.pth")
    sam.to("cuda").eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.7,
                                    stability_score_thresh=0.85, box_nms_thresh=0.7)
    report("+SAM ViT-H")
    sys.path.insert(0, f"{ISM}/../")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14",
                          pretrained=True).cuda().eval()
    report("models loaded")
    def work():
        return len(gen.generate(rgb))

else:
    raise SystemExit(f"unknown role {ROLE}")

# warm + steady loop, holding memory resident for the whole window
n = work(); report("warmup done", f"units={n}")
ts = []
t_hold = time.time()                     # HOLD counts from AFTER load+warmup
while time.time() - t_hold < HOLD:
    s = time.time(); work(); ts.append(time.time() - s)
report("steady", f"n_iter={len(ts)} mean={np.mean(ts):.2f}s")
print(f"[RESULT] role={ROLE} "
      f"torch_peak_alloc={torch.cuda.max_memory_allocated()//2**20} MiB "
      f"torch_reserved={torch.cuda.memory_reserved()//2**20} MiB "
      f"card_now={smi_total()[0]} MiB "
      f"mean_frame={np.mean(ts) if ts else float('nan'):.3f}s n={len(ts)}", flush=True)
