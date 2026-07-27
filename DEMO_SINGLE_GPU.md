# Single-GPU Demo Budget — Measured

Question: can one RTX 4090 (24 GB) hold the four-way segmentation ensemble plus
feature extraction plus registration at once, for a live demo?

**No.** The four routes plus the pose stage need **~37 GB**. Two segmentation
routes plus pose (**21.7 GB**) is the most that fits.

> **Open defect:** two of the four roles were measured proposal-only — their
> DINOv2 pass never ran. The "No" above is unaffected (the undercount only makes
> the total larger), but the *three-way fits* verdict and every latency figure
> are. See [Known measurement defect](#known-measurement-defect-open).

## Instrument

Measured 2026-07-27 on an **RTX 6000 Ada 48 GB** (pod `demo-vram-probe`,
US-IL-1, network volume `8rf4r42sf1`). US-IL-1 had no 4090 capacity that day;
the 6000 Ada is the right substitute because it is **sm_89, the same
architecture as the 4090** — nvdiffrast kernels and the feature caches on the
volume are compiled for sm_89 — with near-identical clocks and bandwidth. VRAM
figures transfer directly; latency is within a few percent. The larger card is
what makes the measurement possible at all: a 24 GB card cannot report the total
of a configuration that does not fit on it.

popoe `d8f98cf` (fresh clone), probe scripts in `scripts/demo_probe/`.

## Per-role footprint, each alone on the card

`peak_alloc` is `torch.cuda.max_memory_allocated` — live tensors, independent of
allocator policy. `nvidia-smi` adds the CUDA context and fragmentation. The two
track closely (MUSE: 7332 vs 7924 − 502 context = 7422), so these are real
requirements, **not** a caching allocator inflating its reserve because the card
is big.

| Role | Models | peak_alloc | reserved | nvidia-smi | Latency |
|---|---|---:|---:|---:|---:|
| MUSE | GroundingDINO-base + SAM2.1 Hiera-L + DINOv2 ViT-g/14-reg | 7332 | 7412 | **7924** | 1.45 s/frame (20 proposals) |
| CNOS-live | SAM2.1-L AMG 32×32 + DINOv2 ViT-g | 7406 ⚠ | 7966 ⚠ | **8478** ⚠ | 1.16 s/frame (50 masks) ⚠ |
| SAM-6D ISM | SAM ViT-H AMG + DINOv2 ViT-L | 6899 ⚠ | 7422 ⚠ | **7946** ⚠ | 1.85 s/frame (177 masks) ⚠ |
| Pose | DINOv2 ViT-g + two-scale GeDi + O3D RANSAC | 5218 | 5320 | **5862** | 1.40 s / 5 detections |

⚠ **These two rows are proposal-only** — the DINOv2 pass their Models column
names was never executed. See [Known measurement defect](#known-measurement-defect-open).

Staged load inside the MUSE process, isolating each weight set:

| Component | MiB |
|---|---:|
| CUDA context | 502 |
| GroundingDINO-base (Swin-B) | 908 |
| SAM2.1 Hiera-L | 994 |
| **DINOv2 ViT-g/14 + registers** | **4324** |
| two-scale GeDi | 24 |

DINOv2 ViT-g is the dominant cost and **every route loads its own copy**. Of the
~37 GB total, roughly 17 GB is four redundant copies of the same 4.3 GB weights.

## Concurrency, measured

| Configuration | Card peak | Verdict on 24 GB |
|---|---:|---|
| CNOS + pose | 14 336 MiB (14.0 GB) | fits, ~10 GB spare |
| MUSE + CNOS + pose | 22 256 MiB (21.7 GB) | fits, 2.3 GB spare ⚠ **at risk** |
| 4 roles, per-process caps summing to 24 GB | — | **all four OOM** |

⚠ This is the one verdict the measurement defect can **flip**: it says "fits",
and its margin is smaller than the workspace CNOS's unexecuted DINOv2 pass would
add. Every other verdict says "does not fit" and an undercount only strengthens
that.

In the capped run every process failed; MUSE died during model loading (5.81 GiB
allowed against 7.4 GiB needed). The sum-of-isolated model is corroborated by
direct measurement: predicted 22 264 MiB for MUSE+CNOS+pose, measured 22 256.

Latency degrades under contention — the routes serialise on the SMs rather than
overlapping:

| Config | MUSE | CNOS | Pose |
|---|---:|---:|---:|
| alone | 1.45 s | 1.16 s | 1.40 s |
| 2-way (CNOS+pose) | — | 2.30 s | 1.80 s |
| 3-way | 2.16 s | 5.07 s | 2.67 s |

## Registration is not the GPU problem

| Solver | Per hypothesis set |
|---|---:|
| `Open3DFeatureRansacSolver` (CPU, headline) | 0.131 s |
| `GPURansacSolver` (10k iters) | 0.007 s |

Registration costs 0.13 s on CPU, 20× less on GPU, and needs no resident weights.
(`registration.ransac_pose_estimation` — the pure-Python reference in
`registration.py` — takes 5.9 s, but the pipeline does not use it.)

Target-side feature encoding is 0.16 s per detection: a 16×16 patch grid inside
the mask, ~60–170 points, one GeDi batch per scale.

## What this means for a demo

Full four-way is off the table on one 24 GB card, on memory *and* on latency:
four routes serial is ~5.9 s of GPU work per frame before pose, and the published
FreeZeV2 figure is ~4.4 s/img (v2.1 Accurate ~24.8 s/img). Options, in order of
leverage:

1. **Run one or two routes.** The repo's own ensemble measurement says the third
   source is worth +0.1 AR on YCB-V — the sources re-propose each other's masks
   (see `ISSUES.md`, 2026-07-16). One route + pose is 14 GB and ~4 s/frame.
2. **Merge into one process with a shared DINOv2.** Removes ~13 GB of duplicate
   weights. Costs the environment isolation the deployment notes deliberately
   keep (SAM-6D pins torch 2.0.0/cu11.3), so it is real work.
3. **fp16 the DINOv2.** Roughly halves its 4.3 GB.
4. **Switch to `GPURansacSolver`** if registration ever lands on the critical
   path — it is not currently the bottleneck.

## Known measurement defect (open)

**Two of the four roles measure only their proposal stage.** `probe_role.py`
loads the DINOv2 descriptor model for `cnos_amg` and `sam6d_ism`, then never
calls it in the timed loop:

| role | `work()` runs | loaded but unused |
|---|---|---|
| `muse` (`probe_role.py:49-54`) | GroundingDINO → SAM2 → `e.embed()` per mask | — complete |
| `pose` (`:84-93`) | `extract_target_features` → `solver.solve` | — complete |
| `cnos_amg` (`:64-65`) | `gen.generate(rgb)` only | `d = DinoV2PatchExtractor(...)` (`:62`) |
| `sam6d_ism` (`:109-110`) | `gen.generate(rgb)` only | `dino = …dinov2_vitl14` (`:106`) |

The defect is not a missing step so much as an **inconsistent table**: the Models
column for both rows explicitly names DINOv2, so they promise a full route and
report a partial measurement, while the MUSE row on the same table is complete.
Nothing distinguishes them to a reader.

**The omission is largest exactly where the table looks best.** Crop counts are
what drive the missing work: MUSE embeds **20** proposals and was measured;
CNOS would embed **50** with the same ViT-g; SAM-6D **177** with a ViT-L. CNOS
is currently the *fastest* row at 1.16 s while carrying 2.5× the crops of the
only row that measured this, so the latency ranking in that table should not be
relied on.

No estimate of the true numbers is offered here on purpose: MUSE's 1.45 s bundles
GroundingDINO + SAM2 + embedding, so a per-crop cost cannot be cleanly recovered
from it. The direction is certain; the magnitude needs the measurement.

### The two risks point in opposite directions

- **VRAM is undercounted, but the headline is safe.** The weights *are* resident
  (visible at the `models loaded` line in every log), so only the forward-pass
  workspace is missing. The headline verdict is "does not fit", and an
  undercount that still does not fit is conservative — **~37 GB is a lower
  bound**.
- **Except for one verdict.** `MUSE + CNOS + pose → fits, 2.3 GB spare` is the
  only row that concludes "fits", and CNOS's missing workspace eats precisely
  that margin. For scale: MUSE's whole work-phase delta — SAM2 refinement *plus*
  20 ViT-g embeds — is `peak_alloc` 6219 → 7316 = **+1097 MiB**. Fifty crops
  plausibly exceed the 2.3 GB headroom. Treat that verdict as unconfirmed.
- **Latency is undercounted in the unsafe direction.** "~5.9 s of GPU work per
  frame before pose" and "One route + pose is 14 GB and ~4 s/frame" (both in
  *What this means for a demo*) are built on these numbers and are optimistic.

### What to do

Run the generated masks through the loaded extractor in both `work()` bodies,
rather than labelling the rows proposal-only — half-route figures cannot answer
this document's question. `muse` already has the pattern (`square_crop` →
`embed`), and `DinoV2ForegroundPatchExtractor.model` is lazily loaded and ready.
Then re-run the isolated and 3-way concurrent probes and update both tables plus
the two latency sentences above.

Worth adding at the same time: an assertion in `work()` that the descriptor
model was actually invoked. `d` and `dino` are currently plain unused locals, so
no linter flags them — which is how this survived.

## Caveats

- **NIDS-Net was not measured**: its repo is not on the network volume, and it
  has only ever been consumed as pre-generated detections JSON. Its stack
  (GroundingDINO + SAM + DINOv2) is MUSE-shaped, so it is estimated at ~7.9 GB.
  The ~37 GB four-way total carries that one estimate.
- The CNOS row is popoe's **live SAM2-AMG surrogate** (`cnos-live`), not the
  official CNOS/FastSAM producer. Official FastSAM is lighter than SAM2-L, so
  that row is likely pessimistic by 1–2 GB.
- Latencies are one YCB-V frame (640×480) on one scene; proposal counts and
  therefore DINOv2 embedding time vary with clutter.
