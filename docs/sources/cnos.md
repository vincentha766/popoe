# CNOS

CNOS is a detection producer. popoe consumes the detections JSON it writes; it does not import the official package. The producer boundary, the shared checks, and `BOPDetectionsSegmentor` usage are in [README.md](README.md).

| Source tag | Meaning |
|---|---|
| `cnos` | Official CNOS / CNOS-FastSAM producer, including the public BOP default detections |
| `cnos-lab` | popoe's local recipe: proposal masks, depth size gate, DINOv2 foreground-patch rank |

Local output never takes the `cnos` tag.

## Upstream

The official `nv-nguyen/cnos` source is pinned as a submodule:

```bash
git submodule update --init --recursive external/cnos
git -C external/cnos rev-parse HEAD
```

The checkout exists for source provenance. It runs in its own environment:

```bash
export POPOE_CNOS_PATH=/path/to/cnos          # defaults to external/cnos
export POPOE_CNOS_PYTHON=/path/to/envs/cnos/bin/python
```

## BOP mode

Prefer the public BOP / CNOS detections. Download and hashes: [`data/detections/cnos/PROVENANCE.md`](../../data/detections/cnos/PROVENANCE.md).

```python
from popoe.segmentor_detections import BOPDetectionsSegmentor

seg = BOPDetectionsSegmentor(
    "data/detections/cnos/cnos-fastsam_lmo-test.json", source="cnos", topk=2
)
```

To regenerate them, build the official command from popoe:

```bash
popoe-cnos infer-command --dataset lmo --model fastsam --rendering-type pbr --gpu 0
```

It prints the equivalent of:

```bash
cd external/cnos && CUDA_VISIBLE_DEVICES=0 python run_inference.py \
  dataset_name=lmo model=cnos_fast model.onboarding_config.rendering_type=pbr
```

The official repo writes BOP-style predictions under its configured Hydra log directory. Validate the result without loading the official environment:

```bash
popoe-cnos check --input data/detections/cnos/cnos-fastsam_lmo-test.json
```

## Custom CAD/RGB mode

The official custom flow is two commands: render templates from a CAD model, then run inference on an RGB image.

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

It writes `OUTPUT_DIR/cnos_results/detection.json` and `vis.png`.

That JSON uses placeholder BOP ids (`scene_id=0`, `image_id=0`, `category_id=1`). Stamp it to the frame and object you will evaluate before consuming it:

```bash
popoe-cnos adapt-custom \
  --input /tmp/cnos_custom/cnos_results/detection.json \
  --output data/detections/cnos/custom_scene0_im42_obj9.json \
  --scene-id 0 --image-id 42 --category-id 9
```

For multi-object output where the JSON already encodes distinct category ids, use `--category-map` with the JSON keys as written — for official custom output that is `1`, not `0`:

```bash
popoe-cnos adapt-custom \
  --input /tmp/cnos_custom/cnos_results/detection.json \
  --output data/detections/cnos/custom_scene0_im42.json \
  --scene-id 0 --image-id 42 --category-map 1:9
```

The adapted JSON keeps `source="cnos"`.

## Local CNOS-lab

`popoe.segmentor_cnos_lab.CNOSLabSegmentor` is the local recipe: proposal masks are filtered by visible 3D extent from depth, then ranked by DINOv2 foreground-patch similarity to templates. It is deliberately separate from official CNOS — write its output under its own path, such as `data/detections/cnos_lab/`, and keep `source="cnos-lab"`.
