# CNOS Deployment Notes

CNOS has two tracks in popoe, and the source names are part of the contract:

| Source | Meaning |
|--------|---------|
| `cnos` | Official CNOS/CNOS-FastSAM producer, including public BOP default detections |
| `cnos-v3` | Local lab recipe: proposal masks -> depth size gate -> DINOv2 foreground-patch rank |
| `cnos-live` | Existing simplified live SAM2+DINOv2 segmentor, not an official result |

Do not write local v3 outputs with `source="cnos"`. Official benchmark numbers
and reproduction headline commands should keep using official/public `cnos`
detections.

## Official Source

The official `nv-nguyen/cnos` source is pinned as a submodule at `external/cnos`:

```bash
git submodule update --init --recursive external/cnos
git -C external/cnos rev-parse HEAD
# 298d1f3366171464ca271659f0e2f7a6eb8e39b4
```

Use this checkout for source provenance and external deployment. popoe consumes
the detections JSON it writes; it does not import the official package.

## Boundary

```text
Official CNOS env/service
  RGB/templates -> 2D detections JSON

popoe env/service
  RGB-D frame manifest + detections JSON -> 6D pose
```

The detections JSON carries only 2D information: `scene_id`, `image_id`,
`category_id`, `score`, `bbox`, and `segmentation`. Depth stays in the frame
manifest and is loaded into `Scene.depth` in metres.

## Environment

The official repository uses Hydra, SAM/FastSAM, DINOv2 and PyTorch packages
that should not be merged into popoe's `pyproject.toml`. Put it in its own
conda/uv environment and point popoe's command builder at it:

```bash
export POPOE_CNOS_PATH=/workspace/cnos
export POPOE_CNOS_PYTHON=/workspace/envs/cnos/bin/python
```

For source-pinned local development, the default path is `external/cnos`.

## BOP Mode

Prefer public BOP/CNOS detections when available:

```python
from popoe.segmentor_cnos_official import CNOSDetectionsSegmentor

seg = CNOSDetectionsSegmentor(
    "data/detections/cnos/cnos-fastsam_lmo-test.json",
    topk=2,
)
```

Equivalent generic union:

```python
from popoe.segmentor_detections import BOPDetectionsSegmentor

seg = BOPDetectionsSegmentor(sources={
    "cnos": "data/detections/cnos/cnos-fastsam_lmo-test.json",
    "sam6d": "data/detections/sam6d_ism_lmo.json",
    "nids": "data/detections/nids/nids_wa_sappe_lmo.json",
}, topk=2)
```

If you need to run the official producer, build the command from popoe:

```bash
popoe-cnos infer-command --dataset lmo --model fastsam --rendering-type pbr --gpu 0
```

It prints a command equivalent to:

```bash
cd external/cnos && CUDA_VISIBLE_DEVICES=0 python run_inference.py \
  dataset_name=lmo model=cnos_fast model.onboarding_config.rendering_type=pbr
```

The official repo writes BOP-style predictions under its configured Hydra log
directory, with filenames based on the segmentor, template rendering type,
aggregation function and dataset. Once a JSON exists, validate it without
loading the official environment:

```bash
popoe-cnos check --input data/detections/cnos/cnos-fastsam_lmo-test.json
```

## Custom CAD/RGB Mode

The official custom flow is two commands: render templates from a CAD model,
then run inference on an RGB image.

```bash
popoe-cnos custom-render-command \
  --cad-path /path/obj.ply \
  --rgb-path /path/rgb.png \
  --output-dir /tmp/cnos_custom

popoe-cnos custom-infer-command \
  --rgb-path /path/rgb.png \
  --output-dir /tmp/cnos_custom \
  --num-max-dets 3 \
  --conf-threshold 0.5 \
  --stability-score-thresh 0.5
```

The official custom script writes:

```text
OUTPUT_DIR/cnos_results/detection.json
OUTPUT_DIR/cnos_results/vis.png
```

If you consume `detection.json` in popoe as official CNOS, use
`source="cnos"`. If you run the local v3 recipe, write it under a separate path
such as `data/detections/cnos_v3/` and keep `source="cnos-v3"`.

## Local CNOS-v3

`popoe.segmentor_cnos_v3.CNOSv3Segmentor` is the local lab recipe migrated from
`gedi/scripts/cnos_match3.py`: proposal masks are filtered by visible 3D extent
from depth, then ranked by DINOv2 foreground-patch similarity to templates.

It is intentionally separate from official CNOS. Use it for real-scene/lab
experiments, not for claiming official CNOS benchmark results.

## Checks

- Confirm the official submodule commit before reproducing results.
- Confirm `source="cnos"` only appears on official/public CNOS files.
- Confirm local v3 outputs use `source="cnos-v3"`.
- Confirm `CNOSSegmentor` live outputs are `source="cnos-live"`.
- Confirm object IDs match the CAD set consumed by popoe.
