#!/usr/bin/env bash
# Boot on pod after a fresh clone of this repo. Records HEAD in the log.
# Usage (on pod):
#   bash scripts/run_fpfh_ablation_pod.sh          # full ycbv+lmo three-arm
#   MODE=pilot bash scripts/run_fpfh_ablation_pod.sh  # LMO objs 1,5,6 only
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=$(git rev-parse --short HEAD)
echo "=== fpfh ablation start $(date -u +%FT%TZ) commit=$COMMIT ==="
echo "hostname=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"

BOP="${BOP:-/workspace/bop_data}"
DET="${DET:-data/detections}"
OUT="${OUT:-/workspace/results/geomablation_20260726}"
export BOP DET OUT

# Prefer existing popoe venv on volume if present
if [ -x /workspace/envs/popoe/bin/python ]; then
  export PATH="/workspace/envs/popoe/bin:$PATH"
  echo "using /workspace/envs/popoe"
elif [ -x /workspace/envs/freezev2/bin/python ]; then
  export PATH="/workspace/envs/freezev2/bin:$PATH"
  echo "using /workspace/envs/freezev2"
fi

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
# detections: use clone's if present, else volume copy
if [ ! -f "$DET/cnos/cnos-fastsam_lmo-test.json" ]; then
  if [ -d /workspace/popoe/data/detections ]; then
    DET=/workspace/popoe/data/detections
  elif [ -d /workspace/detections ]; then
    DET=/workspace/detections
  fi
  export DET
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
