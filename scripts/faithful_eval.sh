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
SEED="${SEED:-42}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Hygiene: leftover tuned env would silently drift a "faithful" run (F6).
unset POPOE_TARGET_GRID POPOE_TARGET_CANON POPOE_TARGET_FILL POPOE_TARGET_CROP
unset POPOE_VIS_DIM POPOE_VIS_WEIGHT POPOE_SKIP_VIS POPOE_DINO_LAYER
unset POPOE_TWO_SCALE_GEDI POPOE_DGEDI_MODE POPOE_GEOM_BACKBONE POPOE_MESH_SHADING
unset POPOE_FPFH_RADII POPOE_FPFH_VOXEL_FRAC POPOE_FPFH_NORMAL_FRAC POPOE_FPFH_ORIENT

export POPOE_CANON_BASIS=diameter
export POPOE_QUERY_POINTS=5000
export POPOE_N_VIEWS=162
export POPOE_QUERY_CANON=476
export POPOE_QUERY_FILL=0.5
export POPOE_QUERY_FILL_MODE=effective
export POPOE_QUERY_MIN_VIEWS=18
export POPOE_QUERY_VIEWS=ico162
export POPOE_TARGET_DENSE=3000
export POPOE_TARGET_PAPER_GRID=1

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

# Refuse to reuse outputs: bop_eval's resume classifies targets by ROW COUNT
# only — pre-fix poses in an old $OUT would be silently kept and the run would
# LOOK like a completed new-code run (audit P1). Fresh OUT dir per identity.
for _f in "$BASE.csv" "${BASE}_cands.csv"; do
  [ -e "$_f" ] && { echo "refusing: $_f exists — resume keeps old rows; use a FRESH OUT dir" >&2; exit 3; }
done

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
  --merge none --topk 2 --grid 16 --solver gpu-feat --seed "$SEED" \
  --weights 1.0 \
  `# Eq.7 coarse term (paper formula, unit exponents per yi-11) — NOT the tuned-line per-dataset S_coarse rule; gedi CONSOLIDATION §2.1` \
  --use-s-coarse \
  --eq5-terms \
  --min-mask-pixels 0 --mask-iou-dedupe 1.1 \
  --tau-diameter \
  --icp-dense --icp-dense-max 3000 \
  --render-backend nvdiffrast \
  --out "$BASE.csv" --cache "$OUT/cache_${TAG}" \
  --cand-csv "${BASE}_cands.csv"

echo "=== VSD render ($TAG) ==="
"$PY" -m popoe.metrics.vsd "$BASE.csv" "$BOP/$BOP_DS" 2>&1 | tail -3
echo "=== flat AR ($TAG) ==="
"$PY" scripts/ar_flat.py "$BASE.csv" "$BOP/$BOP_DS" "faithful/$TAG"
touch "$OUT/DONE_${TAG}"
echo "=== faithful_eval $TAG DONE $(date -u +%FT%TZ) ==="
