#!/usr/bin/env bash
# Geometric-backbone ablation: does the LEARNED geometric branch earn its cost?
#
# The reproduction study's only backbone comparison was GeDi vs dGeDi
# (-16.6 pt on LM-O) — GeDi beating a weaker *learned* descriptor, which does
# not show a learned descriptor is needed at all. This runs the three points
# that do answer it, on identical detections, solver and scoring:
#
#   nogeom  visual-only control      -> is the geometric branch load-bearing?
#   gedi    pure GeDi,     w=0       -> how far does the learned one get alone?
#   fpfh    pure FPFH,     w=0       -> can a free hand-crafted one match it?
#
# and optionally the two fused arms (the headline recipe with each backbone).
#
# Reading the result: fpfh close to gedi reframes the geometric branch as a
# swappable commodity; a wide gedi win justifies the learned branch. Both are
# publishable — Ch3 (backbone justification) / Ch7 (limits).
#
# COST NOTE: nogeom and gedi share one encoder config (the match-time weight is
# swept post-extraction, extraction is pinned at w=1), so with a shared --cache
# dir the second of those two arms costs almost no GPU. Order matters: run
# nogeom or gedi first, then the other, then fpfh.
#
# Usage:
#   BOP=/workspace/bop_data DET=data/detections OUT=out/geomablation \
#     scripts/ablation_geom_backbone.sh
#   DATASETS=lmo ARMS="gedi fpfh" OBJS=--objs=1,5,6 scripts/ablation_geom_backbone.sh
#
# Env:
#   DATASETS  "ycbv lmo"            which datasets to run
#   ARMS      "nogeom gedi fpfh"    add gedi_fused / fpfh_fused for the sweep
#   OBJS      ""                    e.g. --objs=1,5,6 for a cheap pilot
#   FPFH_RADII "0.3,0.4"            canonical-unit radii; matches GeDi r_lrf
set -euo pipefail

BOP="${BOP:?set BOP to the bop_data root}"
DET="${DET:-data/detections}"
OUT="${OUT:-out/geomablation}"
DATASETS="${DATASETS:-ycbv lmo}"
ARMS="${ARMS:-nogeom gedi fpfh}"
OBJS="${OBJS:-}"
FPFH_RADII="${FPFH_RADII:-0.3,0.4}"

mkdir -p "$OUT"
LEDGER="$OUT/ledger.csv"
[ -f "$LEDGER" ] || echo "utc,dataset,arm,backbone,weights,commit,wall_s,csv" > "$LEDGER"

# w=1000 makes the geometric half ~1e-6 of the fused norm: a visual-only control
# without a code change. (A hard zero would need a GEO_WEIGHT knob in fusion.)
VIS_ONLY_W=1000

for ds in $DATASETS; do
  case "$ds" in
    ycbv) DET_ARGS=(--detections "$DET/cnos/cnos-fastsam_ycbv-test.json" --merge ycbv) ;;
    lmo)  DET_ARGS=(--sources "cnos=$DET/cnos/cnos-fastsam_lmo-test.json,sam6d=$DET/sam6d/sam6d_ism_lmo.json" --merge none) ;;
    *) echo "unknown dataset: $ds" >&2; exit 2 ;;
  esac
  # One cache dir per dataset, SHARED across arms — the key already separates
  # backbones (bop_eval enc_cfg), so sharing only ever reuses what is identical.
  CACHE="$OUT/cache_$ds"

  for arm in $ARMS; do
    case "$arm" in
      nogeom)     BACKBONE=gedi; WEIGHTS="$VIS_ONLY_W" ;;
      gedi)       BACKBONE=gedi; WEIGHTS=0.0 ;;
      fpfh)       BACKBONE=fpfh; WEIGHTS=0.0 ;;
      gedi_fused) BACKBONE=gedi; WEIGHTS="1.0,0.7,0.5,0.3,0.2" ;;
      fpfh_fused) BACKBONE=fpfh; WEIGHTS="1.0,0.7,0.5,0.3,0.2" ;;
      *) echo "unknown arm: $arm" >&2; exit 2 ;;
    esac

    CSV="$OUT/${ds}_${arm}.csv"
    LOG="$OUT/${ds}_${arm}.log"
    echo "=== $ds / $arm  (backbone=$BACKBONE weights=$WEIGHTS) -> $CSV"
    start=$(date +%s)
    POPOE_GEOM_BACKBONE="$BACKBONE" POPOE_FPFH_RADII="$FPFH_RADII" \
    uv run python examples/bop_eval.py \
      --bop "$BOP/$ds" \
      "${DET_ARGS[@]}" \
      --topk 2 \
      --grid 32 \
      --solver o3d \
      --weights "$WEIGHTS" \
      --render-backend nvdiffrast \
      --out "$CSV" \
      --cache "$CACHE" \
      --cand-csv "$OUT/${ds}_${arm}_cands.csv" \
      ${OBJS} 2>&1 | tee "$LOG"
    wall=$(( $(date +%s) - start ))

    echo "$(date -u +%FT%TZ),$ds,$arm,$BACKBONE,$WEIGHTS,$(git rev-parse --short HEAD),$wall,$CSV" >> "$LEDGER"
  done
done

cat <<EOF

Done. Ledger: $LEDGER

Next (LOCAL-CPU, no GPU needed):
  1. AR per arm:  python ../gedi/scripts/freezev2_compute_ar_mssd_mspd.py <csv>
     (full BOP AR additionally needs ../gedi/scripts/freezev2_vsd_compute.py)
  2. PAIRED comparison, not just the AR delta — same targets, same detections,
     so build the contingency table (ours-ok/theirs-ok per target) the way the
     FreeZeV2.2 bad-case pass did. A 2-point AR gap with 200 wins and 180
     losses is a different finding from 20 wins and 0 losses.
  3. Per-object breakdown: the dGeDi arm's story was object-dependent
     (obj10 win, obj12 -58.5 pt); expect the same here.
  4. Cost axis from the ledger wall_s column — FPFH is CPU-only, GeDi is not.
  5. MATCHING REGIME is shared on purpose. Both backbones go through the same
     path: fusion L2-normalises the geometric branch, then correspondences are
     nearest-neighbour by cosine. GeDi is metric-trained for that; FPFH is a
     raw histogram. This is the correct controlled comparison (and L2+NN is
     the classical FPFH regime too — PCL/Open3D SAC-IA), so do NOT retune the
     distance for FPFH alone in the headline arm. The one residual: unit-L2
     discards total bin mass, and histogram-native metrics (chi-squared,
     Hellinger, on the RAW 66D) can be slightly better for FPFH. If you want
     that ruled out, add it as a SIDE arm on correspondences only and check
     the GeDi-vs-FPFH ranking is unchanged — a small shift there is a
     suboptimality, not an explanation for a large gap.
  6. THIN OBJECTS need a caveat before you read their per-object numbers.
     FPFH's density normalisation grids the cloud at 4% of object extent, so
     surfaces closer than that merge (scissors blades, clamp jaws). Re-run
     those objects with POPOE_FPFH_VOXEL_FRAC=0.05 or 0 and see whether the
     deficit moves — if it does, that part of it is discretisation, not the
     descriptor. See popoe.descriptors.FPFHDescriptor "Known limitation".
EOF
