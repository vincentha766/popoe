#!/bin/bash
# Emulate a 24 GB RTX 4090: cap each process so the four together get exactly
# what a 4090 has. Caps are proportional to each role's ISOLATED reserved need.
set -u
cd /workspace/probe
export TORCH_HOME=/workspace/torch_cache HF_HOME=/workspace/hf_cache
export POPOE_GEDI_PATH=/workspace/gedi POPOE_SAM2_CKPT=/workspace/sam2_ckpt
R=/workspace/bop_data/ycbv/test/000048/rgb/000001.png
HOLD=${1:-60}
OUT=/workspace/probe_out/cap_$(date +%H%M%S); mkdir -p $OUT

pkill -f 'probe_role[.]py' 2>/dev/null; sleep 5
B=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
echo "card before: ${B} MiB"; [ "$B" -gt 500 ] && { echo REFUSING; exit 1; }

# 24576 MiB total - 4 x 502 MiB CUDA context = 22568 MiB for torch, split
# proportionally to isolated `reserved` (7412 / 7966 / 5320 / 7422 = 28120).
run () {  # role python cap expandable
  local role=$1 py=$2 cap=$3 exp=$4
  if [ "$exp" = 1 ]; then export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  else unset PYTORCH_CUDA_ALLOC_CONF; fi
  $py probe_role.py $role $R $HOLD $cap > $OUT/$role.log 2>&1 &
  echo $!
}
P1=$(run muse      /workspace/envs/muse/bin/python  5949 1)
P2=$(run cnos_amg  /workspace/envs/muse/bin/python  6394 1)
P3=$(run pose      /workspace/envs/dGeDi/bin/python 4270 1)
P4=$(run sam6d_ism /workspace/envs/sam6d/bin/python 5957 0)

echo "ts_s,card_used_MiB,util_pct" > $OUT/smi.csv
T0=$(date +%s)
while kill -0 $P1 $P2 $P3 $P4 2>/dev/null; do
  echo "$(( $(date +%s) - T0 )),$(nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits | tr -d ' ')" >> $OUT/smi.csv
  sleep 2
done
wait 2>/dev/null

echo "======== OUTCOME PER ROLE (4090-sized budget) ========"
for f in $OUT/*.log; do
  r=$(basename $f .log)
  if grep -q 'OutOfMemoryError\|CUDA out of memory' $f; then
    printf '%-12s OOM   <- %s\n' "$r" "$(grep -oE 'Tried to allocate [^;]*|allocate [0-9.]+ [GM]iB' $f | head -1)"
    grep -E '^\[' $f | tail -1
  else
    printf '%-12s OK\n' "$r"; grep -E '^\[RESULT' $f | tail -1
  fi
done
echo "======== CARD ========"
python3 - "$OUT/smi.csv" <<'EOF'
import csv, sys
rows=[r for r in csv.reader(open(sys.argv[1])) if r and r[0]!='ts_s']
m=[int(r[1]) for r in rows]; t=[int(r[0]) for r in rows]
assert t==sorted(t), "multiple monitor writers!"
print(f"samples={len(m)} peak={max(m)} MiB mean={sum(m)//len(m)} MiB window={t[-1]}s")
EOF
echo "artefacts: $OUT"
