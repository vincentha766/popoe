"""Run the OFFICIAL SAM-6D ISM detector under the unified env (torch 2.5.1+cu121).

Mirrors run_inference_custom.py lines 96-165: hydra-compose the official config,
instantiate the official Instance_Segmentation_Model, and run its own segmentor
and descriptor on a real frame. No reimplementation — if this works, the port
holds.

Env recipe (built as /workspace/envs/unified, python 3.10):
    pip install torch==2.5.1 torchvision==0.20.1 --index-url .../cu121
    cp -r <muse-env>/site-packages/sam2 <unified>/site-packages/
    pip install transformers hydra-core omegaconf pytorch-lightning torchmetrics \
                ultralytics trimesh pycocotools distinctipy ruamel.yaml imageio \
                fvcore iopath opencv-python scipy pandas
The official pins (pytorch-lightning==1.8.1, torchmetrics==0.10.3, xformers) are
dropped: lightning is only a base class, and xformers is guarded by try/except.

NOT RUN against current infrastructure. These paths (/workspace/sam2_ckpt,
/workspace/popoe_vramprobe, /workspace/SAM-6D, /workspace/envs/*) are from the
freezev2 volume 8rf4r42sf1, deleted 2026-08-03 (see gedi/RUNPOD.md). Re-point
them at the simtwin volume before trusting any number this prints.
"""
import os, sys, time, subprocess
import numpy as np

ISM = "/workspace/SAM-6D/SAM-6D/Instance_Segmentation_Model"
os.chdir(ISM)
sys.path.insert(0, ISM)

import torch, cv2
# initialize() resolves config_path relative to THIS file; this driver lives
# outside the ISM tree, so use the absolute-dir variant.
from hydra import initialize_config_dir as initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

RGB_PATH = sys.argv[1]
SEGMENTOR = sys.argv[2] if len(sys.argv) > 2 else "sam"

def card():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return int(out.strip().splitlines()[0])

t0 = time.time()
def mark(s):
    print(f"[PORT] {s:34s} card={card():6d} MiB  torch_peak="
          f"{torch.cuda.max_memory_allocated()//2**20:5d} MiB  t={time.time()-t0:6.1f}s", flush=True)

torch.zeros(1, device="cuda")
mark("cuda context")

from hydra.core.global_hydra import GlobalHydra

# The official script nests two initialize() blocks; hydra 1.3 rejects that, so
# compose them one after the other with an explicit clear in between.
with initialize(version_base=None, config_dir=f"{ISM}/configs"):
    cfg = compose(config_name="run_inference.yaml")
GlobalHydra.instance().clear()

cfg.save_dir = "/workspace/probe_out/sam6d_port"
with initialize(version_base=None, config_dir=f"{ISM}/configs/model"):
    cfg.model = compose(config_name="ISM_sam.yaml" if SEGMENTOR == "sam"
                        else "ISM_fastsam.yaml")
GlobalHydra.instance().clear()
if SEGMENTOR == "sam":
    cfg.model.segmentor_model.stability_score_thresh = 0.97

cfg.model.log_dir = cfg.save_dir
os.makedirs(cfg.save_dir, exist_ok=True)

model = instantiate(cfg.model)          # <- the official class
print(f"[PORT] instantiated {type(model).__module__}.{type(model).__name__}", flush=True)

device = "cuda"
model.descriptor_model.model = model.descriptor_model.model.to(device)
model.descriptor_model.model.device = device
if hasattr(model.segmentor_model, "predictor"):
    model.segmentor_model.predictor.model = model.segmentor_model.predictor.model.to(device)
else:
    model.segmentor_model.model.setup_model(device=device, verbose=True)
mark(f"official ISM loaded ({SEGMENTOR})")

rgb = cv2.cvtColor(cv2.imread(RGB_PATH), cv2.COLOR_BGR2RGB)

def one_frame():
    det = model.segmentor_model.generate_masks(np.array(rgb))
    from model.utils import Detections
    det = Detections(det)
    q, qa = model.descriptor_model.forward(np.array(rgb), det)
    # The descriptor pass is the step the earlier probe_role.py roles skipped;
    # assert it produced something so a silent regression cannot recur.
    assert q.ndim == 2 and q.shape[0] > 0, "descriptor pass produced nothing"
    return len(det.masks), tuple(q.shape), tuple(qa.shape)

n, qs, qas = one_frame()
mark("warmup frame")
print(f"[PORT] proposals={n}  descriptors={qs}  appearance={qas}", flush=True)

ts = []
for _ in range(3):
    t = time.time(); one_frame(); ts.append(time.time() - t)
mark("after 3 frames (peak)")
print(f"\n[RESULT] official SAM-6D ISM on torch {torch.__version__}: "
      f"card_peak={card()} MiB torch_peak={torch.cuda.max_memory_allocated()//2**20} MiB "
      f"per_frame={np.mean(ts):.2f}s", flush=True)
