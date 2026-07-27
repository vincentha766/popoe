"""Shared VRAM/latency measurement helpers for the demo-feasibility probe."""
import os, subprocess, time, json, sys

def smi_self():
    """Actual VRAM this process holds, per nvidia-smi (context + weights + workspaces)."""
    pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True)
    except Exception:
        return -1
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0] == str(pid):
            return int(parts[1])
    # RunPod containers usually cannot see per-process rows; we are alone on the
    # card, so whole-card `memory.used` is the same number.
    return smi_total()[0]

def smi_total():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True)
    u, t = [int(x.strip()) for x in out.strip().splitlines()[0].split(",")]
    return u, t

MARKS = []

def mark(label):
    import torch
    torch.cuda.synchronize()
    m = {
        "label": label,
        "smi_proc_MiB": smi_self(),
        "torch_alloc_MiB": round(torch.cuda.memory_allocated() / 2**20),
        "torch_reserved_MiB": round(torch.cuda.memory_reserved() / 2**20),
        "torch_peak_MiB": round(torch.cuda.max_memory_allocated() / 2**20),
    }
    MARKS.append(m)
    print(f"[MARK] {label:38s} smi_proc={m['smi_proc_MiB']:6d} MiB  "
          f"alloc={m['torch_alloc_MiB']:6d}  reserved={m['torch_reserved_MiB']:6d}  "
          f"peak={m['torch_peak_MiB']:6d}", flush=True)
    return m

class timer:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        import torch
        torch.cuda.synchronize(); self.t0 = time.time(); return self
    def __exit__(self, *a):
        import torch
        torch.cuda.synchronize()
        self.dt = time.time() - self.t0
        print(f"[TIME] {self.label:38s} {self.dt:8.2f} s", flush=True)

def dump(path):
    with open(path, "w") as f:
        json.dump(MARKS, f, indent=1)
    print(f"[DUMP] {path}", flush=True)
