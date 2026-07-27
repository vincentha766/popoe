#!/bin/bash
# Measure each role ALONE on the card, reporting torch peak-allocated (the
# allocator-independent requirement) next to nvidia-smi (reserved + context).
set -u
cd /workspace/probe
export TORCH_HOME=/workspace/torch_cache HF_HOME=/workspace/hf_cache
export POPOE_GEDI_PATH=/workspace/gedi POPOE_SAM2_CKPT=/workspace/sam2_ckpt
R=/workspace/bop_data/ycbv/test/000048/rgb/000001.png
HOLD=${1:-30}
CAP=${2:-0}
OUT=/workspace/probe_out/iso_$(date +%H%M%S); mkdir -p $OUT

for role in muse cnos_amg pose sam6d_ism; do
  case $role in
    pose)      PY=/workspace/envs/dGeDi/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ;;
    sam6d_ism) PY=/workspace/envs/sam6d/bin/python; unset PYTORCH_CUDA_ALLOC_CONF ;;
    *)         PY=/workspace/envs/muse/bin/python;  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ;;
  esac
  pkill -f 'probe_role[.]py' 2>/dev/null; sleep 5
  BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  echo "##### ISOLATED $role   (card before: ${BEFORE} MiB)"
  $PY probe_role.py $role $R $HOLD $CAP > $OUT/$role.log 2>&1
  grep -E '^\[' $OUT/$role.log | tail -4
  echo
done
echo "artefacts: $OUT"
