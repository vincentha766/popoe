#!/usr/bin/env bash
# What does rendering the query mesh IN COLOUR buy? One dataset, two arms, one
# commit.
#
# Until 2026-07-28 the renderer decided "is this mesh textured" by asking only
# whether it had a UV atlas + image. BOP ships LM-O, TUD-L, IC-BIN and HB with
# `property uchar red/green/blue` instead, so those four fell through to a flat
# beige Lambertian render and every DINOv2 query feature for them was computed
# on a colourless image — while the target half saw real RGB photographs.
#
# ARM uvonly : POPOE_MESH_SHADING=uv-only, i.e. the old behaviour. It also
#              reproduces the OLD cache keys byte for byte, so it hits the
#              campaign feature cache and costs registration time only.
# ARM vcolor : the fix. New query keys (mesh_shading_key_parts) -> new target
#              keys (they carry the query key) -> full cold re-encode. That is
#              the point: the fix changes pixels, not config, so nothing else
#              in the key would have moved and the wrong features would have
#              been served from cache forever.
#
# Both arms are seeded and run the same selection, so the delta is the features
# and nothing else. --score-coarse also stashes the pre-ICP pose, because this
# bug acts on the FEATURES and should therefore show up in the coarse pose
# first; refinement can mask or amplify it.
#
# Usage (pod, fresh clone, ONE arm per ssh — see RUNPOD.md on truncation):
#   nohup bash scripts/vcolor_ab_run.sh uvonly > /workspace/results/vcolor/uvonly.log 2>&1 &
set -euo pipefail

ARM="${1:?usage: vcolor_ab_run.sh <uvonly|vcolor> [dataset]}"
DS="${2:-lmo}"
OUT="${OUT:-/workspace/results/vcolor_ab_20260728}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/results/pipeline_verify_20260726}"
BOP="${BOP:-/workspace/bop_data}"
DET="${DET:-$PWD/data/detections}"
PY="${PY:-python}"
SEED="${SEED:-1234}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

case "$DS" in
  lmo)
    # The reproduction HEADLINE LM-O run: CNOS u SAM6D union detections, the
    # same configuration the icp_ab A/B used, so its uvonly arm must reproduce
    # those numbers exactly (a free cross-check on the whole harness).
    SRC=(--sources "cnos=$DET/cnos/cnos-fastsam_lmo-test.json,sam6d=$DET/sam6d/sam6d_ism_lmo.json"
         --merge none) ;;
  *) echo "unknown dataset $DS" >&2; exit 2 ;;
esac

case "$ARM" in
  uvonly)
    export POPOE_MESH_SHADING=uv-only
    # The campaign cache. Keys are unchanged under uv-only, so every object
    # must report encode=0.0s; anything else means the key moved and this arm
    # is no longer the old behaviour.
    CACHE="$CACHE_ROOT/cache_${DS}_g32" ;;
  vcolor)
    export POPOE_MESH_SHADING=auto
    # A SEPARATE directory, not the campaign cache: the new entries cannot
    # collide (different keys), but a parallel lane is also using that cache
    # and this arm writes ~1.5 GB into it.
    CACHE="$OUT/cache_${DS}_vcolor" ;;
  *) echo "unknown arm $ARM (uvonly|vcolor)" >&2; exit 2 ;;
esac

mkdir -p "$OUT"
BASE="$OUT/vcolor_${DS}_$ARM"

echo "=== vcolor_ab $DS arm=$ARM shading=$POPOE_MESH_SHADING seed=$SEED cache=$CACHE | $(date -u +%FT%TZ) ==="
# Positive confirmation of WHICH clone is running (the env's editable install
# points somewhere else — see RUNPOD.md's stale-import trap). Printed on
# success too: a check that only speaks on failure cannot prove it ran.
"$PY" -c "import popoe, sys; print('popoe', popoe.__file__)"
BOP_MODELS="$BOP/$DS/$("$PY" -c "from popoe.datasets.bop import bop_layout; print(bop_layout('$DS')['models_dir'])")" \
"$PY" -c "
from popoe.freeze.feature_extractor import resolve_mesh_shading, mesh_shading_key_parts
import trimesh, glob, os
d = os.environ['BOP_MODELS']
fs = sorted(glob.glob(d + '/obj_*.ply'))
modes = {resolve_mesh_shading(trimesh.load(f, force='mesh')) for f in fs}
parts = {mesh_shading_key_parts(f) for f in fs}
print(f'meshes={len(fs)} shading={sorted(modes)} key_parts={sorted(parts)}')
"
git -C "$PWD" rev-parse HEAD | sed 's/^/popoe commit /'

"$PY" examples/bop_eval.py \
  --bop "$BOP/$DS" --dataset "$DS" "${SRC[@]}" \
  --topk 2 --grid 32 --solver o3d --seed "$SEED" \
  --weights 1.0,0.7,0.5,0.3,0.2 \
  --render-backend nvdiffrast \
  --score-coarse \
  --out "$BASE.csv" --cache "$CACHE" --cand-csv "${BASE}_cands.csv"

echo "=== split champion into pre/post-ICP poses ==="
"$PY" scripts/coarse_vs_refined.py \
  --cand-csv "${BASE}_cands.csv" --out-csv "$BASE.csv" --prefix "$BASE"

# VSD needs the GPU rasteriser, so it runs here; MSSD/MSPD are re-derived by
# ar_flat from the same CSV. popoe's scorers now report the BOP flat
# (per-instance) calibre as primary too; ar_flat is kept for the side-by-side
# view of both calibres. vsd.py is run for the .vsd_errs.npz sidecar it
# persists (and its headline AR_VSD is now flat as well).
for V in refined coarse; do
  echo "=== VSD($V) ==="
  "$PY" -m popoe.metrics.vsd "${BASE}_$V.csv" "$BOP/$DS" \
    2>&1 | tee "$OUT/vsd_${DS}_${ARM}_$V.log" | tail -3
  echo "=== AR($V), BOP-flat aggregation ==="
  "$PY" scripts/ar_flat.py "${BASE}_$V.csv" "$BOP/$DS" "$DS $ARM $V" \
    2>&1 | tee "$OUT/ar_${DS}_${ARM}_$V.log"
done

touch "$OUT/DONE_${DS}_$ARM"
echo "=== vcolor_ab $DS $ARM DONE $(date -u +%FT%TZ) ==="
