"""MUSE live route: VRAM + per-frame latency for GroundingDINO + SAM2-L + DINOv2-g."""
import os, sys, time, numpy as np
sys.path.insert(0, "/workspace/popoe_vramprobe/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_common import mark, dump, smi_total

import torch
from dataclasses import dataclass
import cv2

FRAMES = sys.argv[1:]
torch.zeros(1, device="cuda")
mark("00 cuda context only")

from popoe.segmentor_muse import (GroundingDinoBoxProposer, SAM2BoxMaskRefiner,
                                  DinoV2ClsGemEmbedder)
from popoe.segmentor_cnos_v3 import square_crop

@dataclass
class FakeScene:
    rgb: np.ndarray
    depth: np.ndarray = None

prop = GroundingDinoBoxProposer(device="cuda"); prop._load()
mark("01 +GroundingDINO-base")
ref = SAM2BoxMaskRefiner(device="cuda", model_size="large",
                         sam_ckpt_dir="/workspace/sam2_ckpt"); ref._load()
mark("02 +SAM2 Hiera-L")
emb = DinoV2ClsGemEmbedder(device="cuda"); _ = emb.model
mark("03 +DINOv2-g  (ALL THREE RESIDENT)")

def run(path, tag):
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    scene = FakeScene(rgb=rgb)
    torch.cuda.synchronize(); t0 = time.time()
    boxes, _ = prop.propose(scene)
    torch.cuda.synchronize(); t1 = time.time()
    pairs = ref.masks_for_boxes(rgb, boxes)
    torch.cuda.synchronize(); t2 = time.time()
    n = 0
    for m, _iou in pairs:
        crop, cmask = square_crop(rgb, m)
        if crop is None:
            continue
        emb.embed(crop, cmask); n += 1
    torch.cuda.synchronize(); t3 = time.time()
    print(f"[FRAME {tag}] boxes={len(boxes):3d} crops={n:3d} | "
          f"gdino {t1-t0:5.2f}s  sam2 {t2-t1:5.2f}s  dino {t3-t2:5.2f}s  "
          f"TOTAL {t3-t0:5.2f}s", flush=True)
    return t3 - t0

print("\n--- warmup (discarded) ---", flush=True)
run(FRAMES[0], "warmup")
mark("04 after warmup frame (peak)")

print("\n--- steady state ---", flush=True)
ts = [run(p, os.path.basename(os.path.dirname(os.path.dirname(p))) + "/" +
          os.path.basename(p)) for p in FRAMES]
mark("05 after all frames (peak)")

u, tot = smi_total()
print(f"\n[GPU] whole card used {u} / {tot} MiB with the full MUSE route resident")
print(f"[SUM] MUSE steady-state per frame: mean {np.mean(ts):.2f} s "
      f"(min {min(ts):.2f}, max {max(ts):.2f}) over {len(ts)} frames")
dump("/workspace/probe_out/muse_marks.json")
