#!/usr/bin/env bash
# 重建版,以 smoke 验证为准.
#
# Faithful FreeZeV2 A-line wrapper for pod runs. It is intentionally explicit:
# every fidelity knob that changes features or poses is visible either as an
# exported POPOE_* env var or as a bop_eval flag in the command below.
#
# Usage (pod, fresh clone):
#   bash scripts/faithful_eval.sh <lmo|lmo_cnos|ycbv> [<out-root>]
set -euo pipefail

DS="${1:?usage: faithful_eval.sh <lmo|lmo_cnos|ycbv> [<out-root>]}"
OUT="${2:-${OUT:-/workspace/results/faithful_eval}}"
BOP="${BOP:-/workspace/bop_data}"
DET="${DET:-$PWD/data/detections}"
PY="${PY:-python}"
SEED="${SEED:-1234}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export POPOE_CANON_BASIS=diameter
export POPOE_QUERY_POINTS=5000
export POPOE_N_VIEWS=162
export POPOE_QUERY_CANON=476
export POPOE_QUERY_FILL=0.5
export POPOE_QUERY_MIN_VIEWS=18
export POPOE_QUERY_VIEWS=ico162

mkdir -p "$OUT"

case "$DS" in
  lmo|lmo_cnos)
    BOP_DS=lmo
    TAG=lmo_cnos
    DET_JSON="$DET/cnos/cnos-fastsam_lmo-test.json"
    ;;
  ycbv)
    BOP_DS=ycbv
    TAG=ycbv
    DET_JSON="$DET/cnos/cnos-fastsam_ycbv-test.json"
    ;;
  *)
    echo "unknown dataset $DS" >&2
    exit 2
    ;;
esac

BASE="$OUT/faithful_${TAG}"

echo "=== faithful_eval $TAG | seed=$SEED | $(date -u +%FT%TZ) ==="
LOADED=$("$PY" -c "import popoe, os; print(os.path.dirname(popoe.__file__))")
case "$LOADED" in
  "$PWD"/*) echo "popoe import OK: $LOADED" ;;
  *) echo "STALE IMPORT: $LOADED is not under $PWD" >&2; exit 1 ;;
esac
git -C "$PWD" rev-parse HEAD | sed 's/^/popoe commit /'
env | grep '^POPOE_' | sort

"$PY" examples/bop_eval.py \
  --bop "$BOP/$BOP_DS" --dataset "$BOP_DS" \
  --detections "$DET_JSON" \
  --merge none --topk 2 --grid 16 --solver o3d --seed "$SEED" \
  --weights 1.0 \
  --use-s-coarse \
  --corr-topk 10 \
  --tau-diameter \
  --icp-dense --icp-dense-max 3000 \
  --render-rerank \
  --render-backend nvdiffrast \
  --out "$BASE.csv" --cache "$OUT/cache_${TAG}" \
  --cand-csv "${BASE}_cands.csv"

echo "=== VSD render ($TAG) ==="
"$PY" -m popoe.metrics.vsd "$BASE.csv" "$BOP/$BOP_DS" 2>&1 | tail -3
echo "=== flat AR ($TAG) ==="
"$PY" scripts/ar_flat.py "$BASE.csv" "$BOP/$BOP_DS" "faithful/$TAG"
touch "$OUT/DONE_${TAG}"
echo "=== faithful_eval $TAG DONE $(date -u +%FT%TZ) ==="
