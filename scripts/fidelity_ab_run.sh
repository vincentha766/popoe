#!/usr/bin/env bash
# Two FreeZeV2 fidelity fixes, priced separately and together.
#
#   dense — ICP refines against P_T^dense (every valid depth pixel in the mask),
#           not the 32x32 patch-grid cloud. FreeZeV2 Eq. 6 spells the target set
#           P_T^dense; ours was the ~600-point grid cloud, whose own median
#           nearest-neighbour spacing (~2.5 mm) is the same order as tau_ICP.
#   tau   — tau_inlier / tau_ICP / the feature-score inlier radius are 3% of the
#           object's DIAMETER (FreeZeV2 Sec. V-A), not 3% of the sampled query
#           cloud's largest bounding-box side, which runs 2-35% short on LM-O.
#
# Both are pose-side only: neither knob enters `enc_cfg`, so the campaign
# feature caches hit on every target and no GeDi/DINO work is redone. Encode
# times in the log should all read ~0s; anything else is a miss.
#
# The run is seeded (Open3D's RANSAC is otherwise unseeded), and the baseline it
# is compared against is the seed-1234 run of commit c7306a5 in
# /workspace/results/icp_ab_20260728 — same detections, same cache, same solver.
#
# Usage (pod, fresh clone, ONE job per ssh — see RUNPOD.md on truncation):
#   nohup bash scripts/fidelity_ab_run.sh lmo_cnos dense > .../lmo_dense.log 2>&1 &
set -euo pipefail

DS="${1:?usage: fidelity_ab_run.sh <lmo_cnos|lmo|ycbv> <base|dense|tau|both>}"
VAR="${2:?usage: fidelity_ab_run.sh <dataset> <base|dense|tau|both>}"
OUT="${OUT:-/workspace/results/fidelity_ab_20260728}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/results/pipeline_verify_20260726}"
BOP="${BOP:-/workspace/bop_data}"
DET="${DET:-$PWD/data/detections}"
PY="${PY:-python}"
SEED="${SEED:-1234}"
DENSE_MAX="${DENSE_MAX:-3000}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p "$OUT"
BASE="$OUT/fid_${DS}_${VAR}"

BOP_DS="$DS"
case "$DS" in
  lmo_cnos)
    # The SEVEN-SET line's LM-O: CNOS single-source. That is the row the
    # FreeZe(CNOS) comparison is drawn against.
    BOP_DS=lmo
    SRC=(--detections "$DET/cnos/cnos-fastsam_lmo-test.json" --merge none)
    CACHE="$CACHE_ROOT/cache_lmo_g32" ;;
  lmo)
    SRC=(--sources "cnos=$DET/cnos/cnos-fastsam_lmo-test.json,sam6d=$DET/sam6d/sam6d_ism_lmo.json"
         --merge none)
    CACHE="$CACHE_ROOT/cache_lmo_g32" ;;
  ycbv)
    SRC=(--detections "$DET/cnos/cnos-fastsam_ycbv-test.json" --merge ycbv)
    CACHE="$CACHE_ROOT/cache_ycbv_g32m" ;;
  *) echo "unknown dataset $DS" >&2; exit 2 ;;
esac

case "$VAR" in
  base) FIX=() ;;
  dense) FIX=(--icp-dense --icp-dense-max "$DENSE_MAX") ;;
  tau)   FIX=(--tau-diameter) ;;
  both)  FIX=(--icp-dense --icp-dense-max "$DENSE_MAX" --tau-diameter) ;;
  *) echo "unknown variant $VAR" >&2; exit 2 ;;
esac

echo "=== fidelity_ab $DS/$VAR | seed=$SEED | cache=$CACHE | $(date -u +%FT%TZ) ==="
# Positive proof of WHICH popoe ran: the env's editable install points at a
# different checkout, so a bare import would run code this commit does not name.
LOADED=$("$PY" -c "import popoe, os; print(os.path.dirname(popoe.__file__))")
case "$LOADED" in
  "$PWD"/*) echo "popoe import OK: $LOADED" ;;
  *) echo "STALE IMPORT: $LOADED is not under $PWD" >&2; exit 1 ;;
esac
git -C "$PWD" rev-parse HEAD | sed 's/^/popoe commit /'
echo "fixes: ${FIX[*]:-none}"

"$PY" examples/bop_eval.py \
  --bop "$BOP/$BOP_DS" --dataset "$BOP_DS" "${SRC[@]}" \
  --topk 2 --grid 32 --solver o3d --seed "$SEED" \
  --weights 1.0,0.7,0.5,0.3,0.2 \
  --render-backend nvdiffrast \
  "${FIX[@]}" \
  --out "$BASE.csv" --cache "$CACHE"

# VSD is the only leg that needs rendering; it also persists the per-row error
# tensor the flat re-aggregation and the A/B delta both read.
echo "=== VSD render ($DS/$VAR) ==="
"$PY" -m popoe.metrics.vsd "$BASE.csv" "$BOP/$BOP_DS" \
  2>&1 | tail -3

echo "=== flat AR ($DS/$VAR) ==="
"$PY" scripts/ar_flat.py "$BASE.csv" "$BOP/$BOP_DS" "$DS/$VAR"

touch "$OUT/DONE_${DS}_${VAR}"
echo "=== fidelity_ab $DS/$VAR DONE $(date -u +%FT%TZ) ==="
