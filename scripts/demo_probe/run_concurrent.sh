#!/bin/bash
# All four demo roles resident at once; sample whole-card VRAM throughout.
# Refuses to start unless the card is clean, and uses a single monitor writer.
set -u
cd /workspace/probe
export TORCH_HOME=/workspace/torch_cache HF_HOME=/workspace/hf_cache
export POPOE_GEDI_PATH=/workspace/gedi POPOE_SAM2_CKPT=/workspace/sam2_ckpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/bop_data/ycbv/test/000048/rgb/000001.png
HOLD=${1:-120}
ROLES="${2:-muse cnos_amg pose sam6d_ism}"
OUT=/workspace/probe_out/$(date +%H%M%S)
mkdir -p $OUT

pkill -f 'probe_role[.]py' 2>/dev/null; sleep 5
START=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
echo "card at start: ${START} MiB"
if [ "$START" -gt 500 ]; then echo "REFUSING: card not clean"; exit 1; fi

PIDS=""
for role in $ROLES; do
  case $role in
    pose)      PY=/workspace/envs/dGeDi/bin/python; PRE="" ;;
    sam6d_ism) PY=/workspace/envs/sam6d/bin/python
               PRE="env -u PYTORCH_CUDA_ALLOC_CONF" ;;   # torch 2.0 rejects it
    *)         PY=/workspace/envs/muse/bin/python; PRE="" ;;
  esac
  $PRE $PY probe_role.py $role $R $HOLD > $OUT/role_$role.log 2>&1 &
  PIDS="$PIDS $!"
done

echo "ts_s,card_used_MiB,util_pct" > $OUT/smi.csv
T0=$(date +%s)
while kill -0 $PIDS 2>/dev/null; do
  echo "$(( $(date +%s) - T0 )),$(nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits | tr -d ' ')" >> $OUT/smi.csv
  sleep 2
done
wait $PIDS 2>/dev/null

echo "=============== PER-ROLE ==============="
for f in $OUT/role_*.log; do echo "--- $(basename $f)"; grep -E '^\[' $f | tail -3; done
echo "=============== CARD ==================="
python3 - "$OUT/smi.csv" <<'EOF'
import csv, sys
rows=[r for r in csv.reader(open(sys.argv[1])) if r and r[0]!='ts_s']
m=[int(r[1]) for r in rows]; t=[int(r[0]) for r in rows]
assert t==sorted(t), "monitor timestamps out of order -> multiple writers!"
print(f"samples={len(m)}  peak={max(m)} MiB  mean={sum(m)//len(m)} MiB  "
      f"final={m[-1]} MiB  window={t[-1]}s")
EOF
echo "artefacts: $OUT"
