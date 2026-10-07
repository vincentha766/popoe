# MUSE

MUSE is a mask source in FreeZe-v2's segmentation ensemble. The paper is public; the authors' code is not.
There is no official producer to adapt, so popoe includes `popoe.segmentor_muse`, a reimplementation from the paper. This also makes MUSE the only source that acts both as a live segmentor and as its own producer.

The authors' BOP submissions are public, so the artefacts are obtainable even though the producer code is not. The two facts should not be conflated.

The producer boundary, the shared checks, and `BOPDetectionsSegmentor` usage are described in [README.md](README.md).

| Source tag | Meaning |
|---|---|
| `muse` | The authors' official artefacts only. Nothing in popoe writes it. |
| `muse-repro` | This reimplementation (`popoe.segmentor_muse`, `MUSE_SOURCE`) |

Four-way pose comparisons use the authors' official `muse` JSON. Recording a `muse-repro` result under `muse` is a misattribution.

## Upstream

- Paper: [arXiv:2510.17866](https://arxiv.org/abs/2510.17866) (Cho, Park & Oh).
- Code: none published.
- Masks: the authors' public BOP submissions, all seven BOP-Classic-Core sets, saved under `data/detections/muse/` as `muse-full_<ds>-test.json`.
  IDs and SHA256s: [`data/detections/muse/PROVENANCE.md`](../../data/detections/muse/PROVENANCE.md).
  The BOP method page also lists a detection (bbox-only) batch; those files have no masks.
- Training-free by construction: Grounding DINO (Swin-B, prompt "items") → SAM2 (Hiera-L) → DINOv2 template matching (GeM patch + joint score).

## The three paths

```text
popoe env (live)
  RGB-D Scene + CAD templates -> MuseSegmentor -> Detections

popoe env (producer)
  the same Detections -> muse_records / write_muse_detections -> detections JSON

any env (replay)
  detections JSON -> MuseDetectionsSegmentor / BOPDetectionsSegmentor union
```

Use the live path on new captures. Use the exported JSON for evaluation, for multi-source unions, and for any result that must remain reproducible afterwards; replay requires no GPU, no Grounding DINO and no templates.

## Environment

`transformers` (for Grounding DINO) is not a popoe dependency; SAM2 and the DINOv2 hub weights are external as usual.

```bash
pip install -e ".[muse]"                       # torch, transformers, Pillow, pycocotools
pip install git+https://github.com/facebookresearch/sam2.git
export POPOE_SAM2_CKPT=/path/to/sam2_ckpt      # sam2.1_hiera_large.pt lives here
export TORCH_HOME=/path/to/torch_cache         # optional; caches DINOv2 ViT-G
```

Grounding DINO weights are fetched from the HF hub on first use (`IDEA-Research/grounding-dino-base`), so the first run requires network access.

## Register at least two classes

MUSE's relative score is a softmax over all candidate classes, so scoring is joint rather than per-object.
With one registered class that term is the constant 1 and `S_joint` degenerates to `beta * S_abs`.
The library refuses that configuration unless `allow_single_class=True` (CLI: `--allow-single-class`) asks for it explicitly.

This is also why `Segmentor.segment`, a per-object contract, is answered from a `(proposal x class)` score matrix computed once per frame, returning one column per call.

## CLI

Single frame:

```bash
popoe-muse \
  --frame     capture/frame_000000.json \
  --classes   9=/templates/ycbv/obj_000009,14=/templates/ycbv/obj_000014 \
  --models-info /path/to/ycbv/models/models_info.json \
  --out       outputs/muse_ycbv_frame0.json \
  --topk 3
```

The frame manifest is the usual `popoe.datasets.frames` record (`rgb_path`, `depth_path`, `K`, `depth_scale`). Templates are the same rendered PNG directories CNOS-lab uses.

BOP target split:

```bash
popoe-bop-muse \
  --bop-root /path/to/ycbv \
  --template-root /path/to/templates/ycbv \
  --out outputs/muse-repro_ycbv-test.json \
  --shard-dir outputs/muse-repro_ycbv-test_shards \
  --resume
```

`popoe-bop-muse` reads `<bop-root>/test_targets_bop19.json` by default, groups repeated BOP targets so each image is processed once, registers every target object found in the full target file, and writes one combined BOP-format detections JSON.

Flag behaviour to be aware of before interpreting the output:

- `--limit-images` restricts which frames are processed; it does not reduce the registered class set. `--objs 9,14 --limit-images 10` is a reduced-class trial run, and its scores are not comparable with a full multi-class run.
- `--resume` requires `--shard-dir`.
- `--topk` is raised per class to at least that object's BOP `inst_count` on the image, so a target with more instances than `--topk` is not truncated.
- `--target-object-only` filters only the final combined file. Omit it for leaderboard-style segmentation AP.
- The loader expects the usual BOP RGB-D PNG layout (`rgb/` + `depth/`). ITODD's gray `.tif` split needs a dataset-specific loader.

## Library use

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

`Detection.score` is `S_final`, and `Detection.descriptor` holds the breakdown in `DESCRIPTOR_FIELDS` order (`s_abs`, `s_rel`, `p_obj`, `extent_m`). As with every segmentor, that score is comparable only within MUSE.

Proposals are computed once per frame, since Grounding DINO + SAM2 would otherwise be re-run for every object in the image. They are memoised by frame content rather than by `scene_id`/`im_id`, which real captures leave at -1.

### Config identity

`MuseSegmentor.config()` is the per-frame memo key, and the value a `popoe.cache` user should key stored MUSE output on. It covers the class diameters, which determine the size gate, every `DepthSizeGate` field, and each component's settings.

A component may declare its own identity with a `config()` method.
Do this for any custom component holding public mutable state; otherwise reflection includes that state in the key.
Template directories are keyed by path rather than by content, so place edited templates in a new directory instead of modifying the PNGs in place.

Defaults (`alpha=0.5`, `beta=0.8`, `tau=0.02`, `gamma=0.1`, `gem_p=1.5`, prompt `"items."`, GD thresholds 0.15/0.15, SAM2 Hiera-L) match the reference script.

## Divergences from the paper

This is a port of the method rather than a bit-identical replica.

| Paper | Here |
|---|---|
| No depth size gate (BOP proposals are RGB-only) | Depth 3D-extent gate after proposal, over the union of all registered classes' intervals |
| Eqs. (2)–(3): cosine on cls and GeM | Default: cosine(cls) + Tanimoto(GeM); optional `patch_sim=cosine` |
| Naive softmax | Max-subtracted softmax |

The defaults `mask_rgb=True` and `gem_tokens="all"` retain only the object region inside the box, following paper §4.1. Disable them with `--no-mask-rgb --gem-tokens fg`.

Further differences from a typical single-run reference script: the size gate tests the true union of per-class intervals rather than their hull; proposals are memoised per frame because popoe calls `segment` once per object; and models remain resident across frames.

Crop windows also differ slightly from the reference, since `square_crop` here uses an exclusive bbox and PIL BICUBIC, so a small `S_abs` margin should be treated as a tie.

## Verification

`tests/test_segmentor_muse.py` covers the scoring core, cross-class ranking, per-frame memoisation, the size-gate union, and the detections round-trip. GPU-free.
