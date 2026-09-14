#!/usr/bin/env bash
# Geometric-backbone ablation driver. Records HEAD in the log.
# Usage:
#   BOP=/path/to/bop_data OUT=/path/to/out bash scripts/run_fpfh_ablation_pod.sh
#   MODE=pilot BOP=... OUT=... bash scripts/run_fpfh_ablation_pod.sh  # LMO objs 1,5,6
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=$(git rev-parse --short HEAD)
echo "=== fpfh ablation start $(date -u +%FT%TZ) commit=$COMMIT ==="
echo "hostname=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"

BOP="${BOP:?set BOP to the bop_data root}"
DET="${DET:-data/detections}"
OUT="${OUT:?set OUT to the output directory}"
export BOP DET OUT

# Ensure package importable from this clone
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

if [ "${MODE:-full}" = "pilot" ]; then
  export DATASETS=lmo
  export ARMS="nogeom gedi fpfh"
  export OBJS="--objs=1,5,6"
  OUT="${OUT}_pilot"
  export OUT
  echo "MODE=pilot OUT=$OUT OBJS=$OBJS"
else
  export DATASETS="${DATASETS:-ycbv lmo}"
  export ARMS="${ARMS:-nogeom gedi fpfh}"
  export OBJS="${OBJS:-}"
  echo "MODE=full OUT=$OUT DATASETS=$DATASETS ARMS=$ARMS"
fi

mkdir -p "$OUT"
if [ ! -f "$DET/cnos/cnos-fastsam_lmo-test.json" ]; then
  echo "missing $DET/cnos/cnos-fastsam_lmo-test.json — set DET to the detections root" >&2
  exit 1
fi
echo "BOP=$BOP DET=$DET"
ls -la "$BOP/lmo" 2>&1 | head -5
ls -la "$DET/cnos" 2>&1 | head -5

# Sanity: FPFH import
python - <<'PY'
from popoe.descriptors import FPFHDescriptor, FPFH_DIM
print("FPFH ok", FPFH_DIM, FPFHDescriptor)
PY

bash scripts/ablation_geom_backbone.sh
echo "ALL DONE $(date -u +%FT%TZ)" | tee -a "$OUT/DONE"
