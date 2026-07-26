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
  SAM2 (Hiera-L) → DINOv2 template matching with vMF-weighted multi-embedding
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
| vMF-weighted multi-embedding similarity | GeM pooling (p=1.5) + Tanimoto on foreground patch tokens | No per-class concentration parameters to fit from a template set this small |
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
- **Not yet done:** end-to-end parity against `gedi/scripts/muse_match.py` on the
  same frame with the same parameters (masks pixel-identical, scores matching).
  Until that runs on a GPU box, treat any number from this module as unverified
  and do not register it in `REPRODUCTION.md`.
