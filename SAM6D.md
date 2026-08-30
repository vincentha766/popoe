# SAM-6D Deployment Notes

SAM-6D is an external producer for popoe. The official code is pinned as a
submodule at `external/SAM-6D`:

```bash
git submodule update --init --recursive external/SAM-6D
git -C external/SAM-6D rev-parse HEAD
# 1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2
```

Keep this checkout as source provenance and run it in a separate environment or
service. popoe consumes files written by SAM-6D; it does not import the
official package.

## Boundary

```text
SAM-6D ISM env/service
  RGB/templates -> 2D detections JSON

SAM-6D PEM env/service
  RGB-D + CAD + detections/templates -> pose CSV or pose JSON

popoe env/service
  RGB-D frame manifest + detections/pose files -> popoe Detection/PoseHypothesis
```

ISM detections carry only 2D masks, labels, scores and boxes. Depth stays with
`FrameManifest`/`Scene`. PEM translations written in BOP/custom outputs are in
millimetres; popoe converts them to metres when constructing `PoseHypothesis`.

## Environment

The official SAM-6D README reports Python 3.9.6, PyTorch 2.0.0 and CUDA 11.3
for both ISM and PEM. Treat that as incompatible with the lighter popoe env.
On a single 4090, serial execution is the normal shape: run ISM/PEM, let that
process exit and release GPU memory, then run popoe.

Common RunPod volume convention:

```bash
export POPOE_SAM6D_PATH=/workspace/SAM-6D
export POPOE_SAM6D_PYTHON=/workspace/envs/sam6d/bin/python
```

For source-pinned local development, the default path is `external/SAM-6D`.

## BOP ISM

Build the official ISM command from popoe:

```bash
popoe-sam6d ism-command --dataset lmo --model fastsam --gpu 0
```

It prints a command equivalent to:

```bash
cd external/SAM-6D/SAM-6D/Instance_Segmentation_Model
CUDA_VISIBLE_DEVICES=0 python run_inference.py dataset_name=lmo model=ISM_fastsam
```

The resulting `result_<dataset>.json` is consumed as a normal detections file:

```python
from popoe.segmentor_detections import BOPDetectionsSegmentor

seg = BOPDetectionsSegmentor(
    "external/SAM-6D/SAM-6D/Instance_Segmentation_Model/log/sam/result_lmo.json",
    source="sam6d",
    topk=2,
)
```

It is also valid in the generic multi-source union:

```python
from popoe.segmentor_detections import BOPDetectionsSegmentor

seg = BOPDetectionsSegmentor(sources={
    "cnos": "data/detections/cnos/cnos-fastsam_lmo-test.json",
    "sam6d": "external/SAM-6D/SAM-6D/Instance_Segmentation_Model/log/sam/result_lmo.json",
    "nids": "data/detections/nids/nids_wa_sappe_lmo.json",
}, topk=2)
```

## BOP PEM

Build the official PEM command:

```bash
popoe-sam6d pem-command \
  --dataset lmo \
  --checkpoint-path checkpoints/sam-6d-pem-base.pth \
  --gpus 0 \
  --view 42
```

The official script reads its detection paths from `test_bop.py` and documents
that custom segmentation inputs require editing the `detetion_paths` mapping in
that file. Keep such edits in the SAM-6D checkout/env, not in popoe.

PEM writes BOP pose CSV rows. Load them as coarse hypotheses:

```python
from popoe.segmentor_sam6d import SAM6DPemResultsCoarseEstimator

est = SAM6DPemResultsCoarseEstimator(
    "external/SAM-6D/SAM-6D/Pose_Estimation_Model/log/"
    "pose_estimation_model_base_id0/lmo_eval_iter000000/result_lmo.csv",
    topk=1,
)
hyps = est.estimate(scene, obj)   # list[PoseHypothesis], t in metres
```

This is a PEM-results loader, not a `PoseSolver`. It is not wired into the FreeZe
`Pipeline.run` path because PEM already performs a full external pose estimate;
use it as a separate candidate source or service response.

## Real Scene Mode

The official custom demo can write:

```text
OUTPUT_DIR/sam6d_results/detection_ism.json
OUTPUT_DIR/sam6d_results/detection_pem.json
```

Consume `detection_ism.json` with `SAM6DIsmDetectionsSegmentor` if the records
have `scene_id`, `image_id`, `category_id`, `score` and a mask field. Consume
`detection_pem.json` with `SAM6DPemResultsCoarseEstimator` when each record has
`R` and `t`; pass `scene_id`, `image_id` or `obj_id` to the estimator if the
custom output uses missing or placeholder ids (`-1`/`0`). These arguments
stamp ids onto records; they are not filters. If a JSON record already has a
different non-placeholder id, popoe raises instead of silently collapsing it:

```python
est = SAM6DPemResultsCoarseEstimator(
    "captures/frame_000042/sam6d_results/detection_pem.json",
    scene_id=0,
    image_id=42,
    obj_id=9,
)
```

## Checks

- Confirm the SAM-6D submodule commit before reproducing results.
- Confirm ISM masks are present and object ids match popoe's CAD set.
- Confirm PEM `t` is in millimetres before relying on the default
  `translation_scale=0.001`.
- Keep provenance explicit: `source="sam6d"` for ISM detections and
  `source="sam6d-pem"` for PEM pose hypotheses.
