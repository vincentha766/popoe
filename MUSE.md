# MUSE Notes

MUSE is the fourth mask source in FreeZeV2's segmentation ensemble, and the odd
one out: **there is no official code and no downloadable masks.** So popoe
cannot adapt an official producer the way it does for CNOS, SAM-6D and NIDS-Net.
What it has instead is `popoe.segmentor_muse` — a reimplementation from the
paper.

## Source Names

| Source | Meaning |
|--------|---------|
| `muse` | RESERVED for official MUSE artefacts. **Nothing in popoe writes it.** |
| `muse-repro` | This reimplementation (`popoe.segmentor_muse`, `MUSE_SOURCE`) |

This split matters more here than anywhere else in the repo. The reproduction
study cites MUSE as evidence that one of FreeZeV2's four ensemble members is
externally unreproducible; a number produced by our own reimplementation, filed
under the official name, would quietly refute an argument the study is making on
purpose. Do not relabel.

## Upstream Status

- Paper: [arXiv:2510.17866](https://arxiv.org/abs/2510.17866) (Oct 2025), Cho,
  Park & Oh — an independent team, **not** the FreeZe/FBK group.
- Code: none published. Masks: none downloadable (BOP `method_info/873`).
- Training-free by construction: Grounding DINO (Swin-B, prompt "items") →
  SAM2 (Hiera-L) → DINOv2 template matching (GeM patch + joint score)
  similarity.
- FreeZeV2 consumes it as a pure upstream data dependency; BOP rules only
  require *declaring* the segmentation source, not publishing it.

## Boundary — Both Halves Live Here

Unlike the other three backends, MUSE is not "external producer + popoe
adapter". It is a live segmentor *and* its own producer:

```text
popoe env (live)
  RGB-D Scene + CAD templates -> MuseSegmentor -> Detections

popoe env (producer)
  the same Detections -> muse_records/write_muse_detections -> detections JSON

any env (replay)
  detections JSON -> MuseDetectionsSegmentor / BOPDetectionsSegmentor union
```

Use the live path on new captures; use the dumped JSON for evaluation, for
multi-source unions, and for anything that must stay reproducible after the
fact. The replay path needs no GPU, no Grounding DINO and no templates.

## Environment

`transformers` (for Grounding DINO) is not a popoe dependency; SAM2 and the
DINOv2 hub weights are external as usual.

```bash
pip install -e ".[muse]"                       # torch, transformers, Pillow, pycocotools
pip install git+https://github.com/facebookresearch/sam2.git
export POPOE_SAM2_CKPT=/workspace/sam2_ckpt    # sam2.1_hiera_large.pt lives here
export TORCH_HOME=/workspace/torch_cache       # avoids re-downloading DINOv2 ViT-G
```

Grounding DINO weights come from the HF hub on first use
(`IDEA-Research/grounding-dino-base`), so the first run needs network access.

## CLI

Single frame:

```bash
popoe-muse \
  --frame     capture/frame_000000.json \
  --classes   9=/templates/ycbv/obj_000009,14=/templates/ycbv/obj_000014 \
  --models-info /workspace/bop_data/ycbv/models/models_info.json \
  --out       outputs/muse_ycbv_frame0.json \
  --topk 3
```

The frame manifest is the usual `popoe.datasets.frames` record
(`rgb_path`, `depth_path`, `K`, `depth_scale`). Templates are the same rendered
PNG directories CNOS-v3 uses. The command prints the surviving-proposal count
and each class's best score breakdown, writes the detections JSON, then reloads
it through `load_bop_detections` so a schema error surfaces immediately.

**Pass at least two classes.** MUSE's relative score is a softmax across
classes; with one registered class it is the constant 1 and `S_joint`
degenerates to `beta * S_abs`. The library refuses that configuration unless
`allow_single_class=True` (CLI: `--allow-single-class`) asks for it explicitly.

BOP target split:

```bash
popoe-bop-muse \
  --bop-root /workspace/bop_data/ycbv \
  --template-root /workspace/templates/ycbv \
  --out outputs/muse-repro_ycbv-test.json \
  --shard-dir outputs/muse-repro_ycbv-test_shards \
  --resume
```

`popoe-bop-muse` reads `<bop-root>/test_targets_bop19.json` by default, groups
repeated BOP targets so each image is processed once, registers every target
object found in the full target file, and writes one combined BOP-format
detections JSON. `--limit-images` only limits which frames are processed; it
does not shrink the registered MUSE class set, because MUSE's relative score is
a softmax across that set. Use `--objs 9,14 --limit-images 10` when you want a
reduced-class smoke run; those scores are not directly comparable with a full
multi-class run. Per-image shards are full registered-class output and are keyed
by dataset root, split, MUSE config, class set and effective per-class `topk`;
`--resume` therefore requires `--shard-dir` and will not reuse shards from a
different smoke/full configuration. `--topk` is floored per class by that
object's BOP `inst_count` on the image. `--target-object-only` only filters the
final combined file. Leave that flag off for leaderboard-style segmentation AP,
where wrong-category detections on target images should remain false positives.
`--no-time` strips runtime from the combined output only; shards still keep time
so resumed default runs can preserve it. For
sensor-named BOP test splits such as `test_primesense`, the default targets path
falls back to `test_targets_bop19.json`. The loader expects the usual BOP RGB-D
PNG layout (`rgb/` + `depth/`); ITODD's gray `.tif` split needs a
dataset-specific loader before this CLI can cover it.

## Library Use

```python
from popoe.interfaces import ObjectModel
from popoe.segmentor_muse import MuseClass, build_muse_segmentor, muse_records

seg = build_muse_segmentor([
    MuseClass(ObjectModel(9, ".../obj_000009.ply", diameter=0.130), "/templates/obj_000009"),
    MuseClass(ObjectModel(14, ".../obj_000014.ply", diameter=0.125), "/templates/obj_000014"),
])
dets = seg.segment(scene, obj)          # Detection.source == 'muse-repro'
records = muse_records(scene, seg)      # the same masks, as a detections JSON
```

`Detection.score` is `S_final`; `Detection.descriptor` carries the breakdown in
`DESCRIPTOR_FIELDS` order (`s_abs`, `s_rel`, `p_obj`, `extent_m`). As with every
segmentor, that score is comparable only within MUSE.

Proposals are computed once per **frame** (Grounding DINO + SAM2 would otherwise
re-run for every object in the image) and memoised by frame CONTENT, not by
`scene_id`/`im_id` — real captures frequently leave those at -1, and reusing one
frame's masks on another is exactly the stale-mask failure seen on the
2026-07-24 real-shot run.

## Divergences From the Paper

Carried over from the validated script `gedi/scripts/muse_match.py`. Keep this
list honest — it is what stops these numbers being read as an exact replication.

| Paper | Here | Why |
|-------|------|-----|
| No depth size gate (BOP proposals are RGB-only) | Depth 3D-extent gate after proposal, union of all registered classes' intervals | Cheap, already validated on this project's real shots, targets the printed-text confuser failure mode |
| Paper Eqs. (2)–(3): cosine on cls **and** GeM | Default: cosine(cls) + **Tanimoto**(GeM); optional ``patch_sim=cosine`` | G2 A/B; paper prose also names Tanimoto |
| — | No union-bbox box-prompt refinement (CNOS-v3 has one) | The GD+SAM2 cascade is the paper's largest ablation lever (+0.108 mAP over plain SAM proposals); it should not need the patch |
| Naive softmax | Max-subtracted softmax | Identical on any input the reference handles; additionally keeps an all-failed proposal row from becoming NaN and poisoning every class's ranking |

Two departures from the reference script itself (not from the paper), both
tightening behaviour rather than changing the method:

- The size gate tests the **true union** of the per-class intervals, where the
  script used their hull `[min(lo_c), max(hi_c)]`. Identical whenever the
  registered diameters span less than 4.4x (all YCB-V/LM-O pairs used so far);
  beyond that the hull admits proposals plausible for no class.
- Proposals are memoised per frame, because popoe calls `segment` once per
  object while the script processed all classes in one pass.
- **Models stay resident.** The script loads Grounding DINO, frees it
  (`del gdino; torch.cuda.empty_cache()`), loads SAM2, frees that too, then
  matches — at most one heavy model in VRAM at a time. A live segmentor serves
  many frames, so all three (GD ~0.7 GB + SAM2-L ~0.9 GB + DINOv2 ViT-G ~4.5 GB)
  stay loaded. Comfortable on a 24 GB 4090; on a smaller card the script would
  fit where this does not.

### Config identity

`MuseSegmentor.config()` is the per-frame memo key and the handle a
`popoe.cache` user should key stored MUSE output on. It covers class diameters
(they drive the size gate), every `DepthSizeGate` field, and each component's
settings. A component may declare its own identity with a `config()` method —
**do this for any custom component holding public mutable state** (a counter, a
handle), or reflection will fold that state into the key and every frame will
miss the memo. Template directories are keyed by path, not content: point a new
directory at edited templates rather than editing PNGs in place.

Defaults (`alpha=0.5`, `beta=0.8`, `tau=0.02`, `gamma=0.1`, `gem_p=1.5`, prompt
`"items."`, GD thresholds 0.15/0.15, SAM2 Hiera-L) match the reference script.

## Verification Status

- Unit tests: `tests/test_segmentor_muse.py` — scoring core, cross-class
  ranking, per-frame memoisation, size-gate union, detections round-trip. GPU-free.
- End-to-end parity against `gedi/scripts/muse_match.py`: **run 2026-07-26** on a
  4090, same frame (`shots_0723/rgb_000000.png`), same parameters, popoe commit
  `e07a129`. Artefacts: `gedi/muse_parity_20260726/`.

| check | result |
|---|---|
| proposals surviving the size gate | 3 of 7, **both** |
| mask SET produced (content-aligned) | **identical**, 3 of 3 |
| same mask at every rank | **no** — `obj_000009`'s top-1/top-2 swapped |
| max abs delta on `S_final` | 0.0168 |

**Read this as: the port reproduces the method, not the arithmetic.** Proposal
and gating are bit-identical — Grounding DINO, SAM2 and the depth gate select
exactly the same three regions. Every difference is in scoring, and it comes
from the two `square_crop` implementations, which are not the same function:

| | reference (`cnos_match3.py:31`) | popoe (`segmentor_cnos_v3.py:41`) |
|---|---|---|
| bbox | inclusive (`ys.max()`) | exclusive (`ys.max() + 1`) |
| half-width | `(hw + int(hw*2*pad)) // 2` | `round(side * (0.5 + pad))` |
| resample | cv2 `INTER_LINEAR` | PIL `BICUBIC` |

Measured on this frame's masks, the crop windows differ by 2 px in size and
1 px in origin, which is enough to move `S_abs` by 0.003–0.017.

`obj_000009`'s top-1 and top-2 sat 0.0031 apart in the reference run — below
that sensitivity — so the ranking flipped. `obj_000014`, whose margin was
0.0496, ranked identically. **Neither implementation is "right" here**: a
0.003 margin means this frame does not discriminate between those two masks at
all, in either version. Treat a sub-0.02 margin on this pipeline as a tie, not
a decision.

**Decision (2026-07-26): bit-parity is not pursued.** popoe's exclusive bbox is
the geometrically correct one — the reference's inclusive bbox is an off-by-one
— so aligning would mean freezing that off-by-one into this repo. The shared
`square_crop` also serves CNOS-v3, an evaluated path, so changing it needs its
own A/B rather than a drive-by edit. What matters is recorded instead: the two
crop conventions differ, the resulting score sensitivity is ~0.017, and any
margin below that is a tie.

Nothing has been registered in `REPRODUCTION.md`: this run verifies the port
against its reference, it does not produce a benchmark number.

## G1 result (2026-07-26) — depth size gate A/B

LM-O full split, topk=3, same templates/env as campaign2 muse-repro.
Code: branch `g1-muse-depth-gate-ab` @ `8f7378d`, flag `--no-size-gate`.
Pod: reused `ctxs25f2eczigt` (network volume). Artefacts:
`outputs/g1_muse_gate_20260726/`, pod
`/workspace/results/g1_muse_gate_20260726/`.

| condition | AP | AP50 | AP75 |
|---|---:|---:|---:|
| official `muse` | **0.471** | — | — |
| muse-repro **gate on** (default) | **0.228** | 0.377 | 0.259 |
| muse-repro **gate off** (G1) | **0.224** | 0.376 | 0.243 |

**Conclusion:** disabling the depth 3D-extent gate does **not** close the
official gap (Δ ≈ −0.004 vs gate-on, noise-level). Next levers: **G2**
**G2** patch_sim cosine (paper eqs) vs Tanimoto, then **G3** GD/SAM2 cascade.
Keep the gate available for real-robot confuser filtering; it is not the
BOP AP bottleneck.
