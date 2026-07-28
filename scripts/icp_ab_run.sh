#!/usr/bin/env bash
# What does ICP refinement buy us? One dataset, end to end, on the pod.
#
# FreeZe's Table 3 credits ICP with +4.76 pt mean AR. Our seven-set line already
# includes ICP and still lands below FreeZe's NO-refinement row, which leaves two
# incompatible readings: either our ICP is worth about as much as theirs (so the
# coarse pose feeding it is far worse than theirs — a features/RANSAC problem),
# or our ICP is worth nearly nothing (an ICP problem). Those need opposite fixes,
# so measure before choosing.
#
# This runs the pipeline UNCHANGED with --score-coarse, which stashes each
# hypothesis' pre-ICP pose, then scores the champion's pre- and post-ICP poses
# against the same ground truth. See scripts/coarse_vs_refined.py for what that
# does and does not measure (it is not "the pipeline without refinement").
#
# The run is seeded: the two rows must come from ONE run to be comparable, and
# Open3D's RANSAC is otherwise unseeded, so an unseeded re-run of the same config
# lands somewhere else. That also makes this experiment repeatable — and means it
# is NOT bit-comparable to the unseeded seven-set line.
#
# Registration-only: pose knobs are absent from the encoder cache key, so a
# campaign feature cache hits on every target and no GeDi/DINO work is redone.
# Encode times in the log should all be ~0s; anything else means a miss, and
# misses here are recomputed by newer code than built the cache.
#
# Usage (pod, fresh clone, ONE dataset per ssh — see RUNPOD.md on truncation):
#   nohup bash scripts/icp_ab_run.sh lmo  > /workspace/results/icp_ab/lmo.log 2>&1 &
#   nohup bash scripts/icp_ab_run.sh ycbv > /workspace/results/icp_ab/ycbv.log 2>&1 &
set -euo pipefail

DS="${1:?usage: icp_ab_run.sh <lmo|lmo_cnos|ycbv|tudl>}"
OUT="${OUT:-/workspace/results/icp_ab_20260728}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/results/pipeline_verify_20260726}"
BOP="${BOP:-/workspace/bop_data}"
DET="${DET:-$PWD/data/detections}"
PY="${PY:-python}"
SEED="${SEED:-1234}"
# Open3D RANSAC is the bottleneck and it is CPU-bound; uncapped it thrashes when
# two datasets share a box (campaign note, 2026-07-26).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p "$OUT"
BASE="$OUT/icp_ab_$DS"

BOP_DS="$DS"
case "$DS" in
  lmo)
    # The reproduction HEADLINE LM-O run: CNOS u SAM6D union detections.
    SRC=(--sources "cnos=$DET/cnos/cnos-fastsam_lmo-test.json,sam6d=$DET/sam6d/sam6d_ism_lmo.json"
         --merge none)
    CACHE="$CACHE_ROOT/cache_lmo_g32" ;;
  lmo_cnos)
    # The SEVEN-SET line's LM-O (0.631) is CNOS single-source, and that is the
    # row the FreeZe Table 3 comparison is drawn against — the union run scores
    # ~5 pt higher and would answer a different question. Same cache: with
    # topk=2 the CNOS masks here are the union run's CNOS bucket, and cache keys
    # are per (scene, mask, object), so every target still hits.
    BOP_DS=lmo
    SRC=(--detections "$DET/cnos/cnos-fastsam_lmo-test.json" --merge none)
    CACHE="$CACHE_ROOT/cache_lmo_g32" ;;
  ycbv)
    SRC=(--detections "$DET/cnos/cnos-fastsam_ycbv-test.json" --merge ycbv)
    CACHE="$CACHE_ROOT/cache_ycbv_g32m" ;;
  tudl)
    # No campaign cache exists for TUD-L: this branch pays full feature
    # extraction and is only worth starting with hours of budget in hand.
    SRC=(--detections "$DET/cnos/cnos-fastsam_tudl-test.json" --merge none)
    CACHE="$OUT/cache_tudl_g32" ;;
  *) echo "unknown dataset $DS" >&2; exit 2 ;;
esac

echo "=== icp_ab $DS | seed=$SEED | cache=$CACHE | $(date -u +%FT%TZ) ==="
"$PY" -c "import popoe, sys; print('popoe', popoe.__file__)"
git -C "$PWD" rev-parse HEAD | sed 's/^/popoe commit /'

"$PY" examples/bop_eval.py \
  --bop "$BOP/$BOP_DS" --dataset "$BOP_DS" "${SRC[@]}" \
  --topk 2 --grid 32 --solver o3d --seed "$SEED" \
  --weights 1.0,0.7,0.5,0.3,0.2 \
  --render-backend nvdiffrast \
  --score-coarse \
  --out "$BASE.csv" --cache "$CACHE" --cand-csv "${BASE}_cands.csv"

echo "=== split champion into pre/post-ICP poses ==="
"$PY" scripts/coarse_vs_refined.py \
  --cand-csv "${BASE}_cands.csv" --out-csv "$BASE.csv" --prefix "$BASE"

# Full BOP AR = mean(MSSD, MSPD, VSD), the same three legs FreeZe Table 3 uses.
for V in refined coarse; do
  echo "=== AR($V) ==="
  BOP_PATH="$BOP/$BOP_DS" "$PY" -m popoe.metrics.ar "${BASE}_$V.csv" \
    2>&1 | tee "$OUT/ar_${DS}_$V.log" | tail -5
  echo "=== VSD($V) ==="
  "$PY" -m popoe.metrics.vsd "${BASE}_$V.csv" "$BOP/$BOP_DS" \
    2>&1 | tee "$OUT/vsd_${DS}_$V.log" | tail -3
done

touch "$OUT/DONE_$DS"
echo "=== icp_ab $DS DONE $(date -u +%FT%TZ) ==="
