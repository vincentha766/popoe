#!/usr/bin/env bash
# The FAITHFUL configuration: every knob at the FreeZeV2 paper value.
#
# SCOPE: faithful-cnos ONLY (CNOS-FastSAM single source, LM-O or YCB-V).
# The 18-run ledger entries (faithful-4way / tuned-*) live as bash blocks in
# REPRODUCTION.md — do not treat this wrapper as a substitute for those arms.
#
# This is experiment A of the fidelity programme: one number per dataset for
# "popoe configured as the paper describes", to stand against the official
# FreeZe(CNOS) rows. The tuned line (grid 32, weight sweep, champion rule) is
# a DIFFERENT line and is not touched by this script.
#
# Knob -> ledger row (BOP_OFFICIAL_BASELINES.md 配置对账总表):
#   --grid 16                      #10  target grid 16x16 (<=256 patches)
#   --weights 1.0                  #11  fixed fusion weight, no sweep
#   --use-s-coarse + --merge none  #12  score = s_icp * s_feat_1 * s_coarse
#                                       (= Eq. 7 at unit exponents; metric_fit
#                                       is inert without merge pooling)
#   POPOE_QUERY_MIN_VIEWS=18       #14  visibility gate, >=18 of 162 views
#   POPOE_CANON_BASIS=diameter     #15  GeDi radii vs the object DIAMETER
#   POPOE_QUERY_POINTS=5000        #16  |P_Q| = 5k
#   POPOE_QUERY_CANON=476          #17  render canvas (34*14; the paper's 480
#   POPOE_QUERY_FILL=0.5                is not a multiple of the 14px patch)
#   --solver gpu-feat              #18  Eq.5 selection; native k=10 (o3d-only
#                                       --corr-topk must NOT be passed here)
#   --tau-diameter                 # 8  tau = 3% of the diameter (PR #20)
#   --icp-dense --icp-dense-max 3000  # 7  ICP against P_T^dense, 3k points
#   --render-rerank --render-score    v2.1 (decision 13): SAR≈rerank + the
#                                       render-vs-input appearance score in
#                                       champion selection (pinned-by-us
#                                       approximations of Row 19 prose)
#   --mask-m 2n                       v2.1 (decision 13): mask budget floor
#                                       "up to M = 2N"; at N=1 identical to
#                                       the paper's N+1
#
# Exponents alpha/beta/gamma are UNPUBLISHED (ledger #19); unit exponents are
# our choice and the offline sweep bounds the difference at <= +0.24pt (乙-7).
#
# Every feature-side knob above is non-default, so enc_cfg keys change and the
# cache is cold BY CONSTRUCTION — this run cannot be served the tuned line's
# features. Budget accordingly (full query + target extraction).
#
# Usage (pod, fresh clone, one job per ssh):
#   bash scripts/faithful_run.sh <lmo_cnos|ycbv> [<out-root>]
set -euo pipefail

DS="${1:?usage: faithful_run.sh <lmo_cnos|ycbv> [<out-root>]}"
OUT="${2:-${OUT:-/workspace/results/faithful_20260728}}"
BOP="${BOP:-/workspace/bop_data}"
DET="${DET:-$PWD/data/detections}"
PY="${PY:-python}"
SEED="${SEED:-42}"     # runbook pin; 1234 was pre-freeze wrapper drift (F6)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Hygiene: a shell with leftover tuned env would silently drift a "faithful"
# run (enc_cfg records it but does not refuse it) — unset like the runbooks.
unset POPOE_TARGET_GRID POPOE_TARGET_CANON POPOE_TARGET_FILL POPOE_TARGET_CROP
unset POPOE_VIS_DIM POPOE_VIS_WEIGHT POPOE_SKIP_VIS POPOE_DINO_LAYER
unset POPOE_TWO_SCALE_GEDI POPOE_DGEDI_MODE POPOE_GEOM_BACKBONE POPOE_MESH_SHADING
unset POPOE_FPFH_RADII POPOE_FPFH_VOXEL_FRAC POPOE_FPFH_NORMAL_FRAC POPOE_FPFH_ORIENT

export POPOE_QUERY_POINTS=5000
export POPOE_N_VIEWS=162
export POPOE_QUERY_VIEWS=ico162
export POPOE_QUERY_CANON=476
export POPOE_QUERY_FILL=0.5
export POPOE_QUERY_FILL_MODE=effective
export POPOE_QUERY_MIN_VIEWS=18
export POPOE_CANON_BASIS=diameter
export POPOE_TARGET_DENSE=3000
export POPOE_TARGET_PAPER_GRID=1

mkdir -p "$OUT"
BASE="$OUT/faithful_${DS}"

# Refuse to reuse outputs (audit P1): resume classifies by ROW COUNT only —
# an old $OUT would silently keep pre-fix poses and look freshly run.
for _f in "$BASE.csv" "$BASE.cand.csv" "$BASE.csv.vsd_errs.npz" "$OUT/DONE_${DS}"; do
  [ -e "$_f" ] && { echo "refusing: $_f exists — resume keeps old rows (and stale vsd/DONE artifacts read as fresh); use a FRESH OUT dir" >&2; exit 3; }
done

# Rerank file guard (A4; back with decision 13 — faithful carries rerank again):
for req in scripts/check_rerank_symmetry.py scripts/sar_render_compare.py; do
if [ ! -f "$req" ]; then
  echo "missing $req — stale tag tip (render_rerank loads sar_render_compare at RUNTIME); fetch --tags --force" >&2
  exit 1
fi
done
export POPOE_GEDI_PATH="${POPOE_GEDI_PATH:-/workspace/gedi}"
export POPOE_BOP_TOOLKIT="${POPOE_BOP_TOOLKIT:-/workspace/bop_toolkit}"

case "$DS" in
  lmo_cnos) BOP_DS=lmo
    SRC=(--detections "$DET/cnos/cnos-fastsam_lmo-test.json") ;;
  ycbv) BOP_DS=ycbv
    SRC=(--detections "$DET/cnos/cnos-fastsam_ycbv-test.json") ;;
  *) echo "unknown dataset $DS" >&2; exit 2 ;;
esac

echo "=== faithful $DS | seed=$SEED | $(date -u +%FT%TZ) ==="
LOADED=$("$PY" -c "import popoe, os; print(os.path.dirname(popoe.__file__))")
case "$LOADED" in
  "$PWD"/*) echo "popoe import OK: $LOADED" ;;
  *) echo "STALE IMPORT: $LOADED is not under $PWD" >&2; exit 1 ;;
esac
git -C "$PWD" rev-parse HEAD | sed 's/^/popoe commit /'
env | grep '^POPOE_' | sort

"$PY" examples/bop_eval.py \
  --bop "$BOP/$BOP_DS" --dataset "$BOP_DS" "${SRC[@]}" \
  --merge none --topk 2 --grid 16 --solver gpu-feat --seed "$SEED" \
  --weights 1.0 \
  --use-s-coarse \
  --eq5-terms \
  --min-mask-pixels 0 --mask-iou-dedupe 1.1 \
  --tau-diameter \
  --trans-nms 0.05 \
  --icp-dense --icp-dense-max 3000 \
  --render-rerank --render-score --mask-m 2n \
  --render-backend nvdiffrast \
  --out "$BASE.csv" --cache "$OUT/cache_${DS}" \
  --cand-csv "$BASE.cand.csv"

echo "=== VSD render ($DS) ==="
"$PY" -m popoe.metrics.vsd "$BASE.csv" "$BOP/$BOP_DS" 2>&1 | tail -3
echo "=== flat AR ($DS) ==="
BOP_PATH="$BOP/$BOP_DS" "$PY" -m popoe.metrics.ar "$BASE.csv"
touch "$OUT/DONE_${DS}"
echo "=== faithful $DS DONE $(date -u +%FT%TZ) ==="
