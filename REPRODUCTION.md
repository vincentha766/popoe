# REPRODUCTION — parity ledger vs the `gedi` archive

The reproduction study lives in the frozen archive repo (`../gedi`, see its
`EXPERIMENTS.md` / `DISSERTATION_PLAN.md`). Before the dissertation cites
popoe-produced numbers, every headline result must be re-run through popoe
entrypoints and logged here. Until a row is checked, the gedi number remains
the authoritative (historical) figure.

**Acceptance rule**: same BOP test split, same recipe, full BOP AR within
±0.003 of the archive number (RANSAC stochasticity; tighten to bit-identical
if the run is seeded). Record the popoe commit hash for every reproduced
number.

## Context: the popoe promotion line already exists

popoe's own union2 + S_coarse campaign (2026-07-17 full-set run, 07-21 grasp
follow-up) already produced a popoe-native formal line: YCB-V full BOP AR
**0.8201**, LM-O **0.6896**, grasp ADD(-S)@0.1d 0.8616 / 0.7816. Artifacts:
`../gedi/ycbv_local_data/union_scoring_20260716/`. That line needs no parity
run — it was born on popoe. The ledger below is about re-producing the **gedi
script line** (the dissertation's reproduction headline) through popoe
entrypoints, under the dual-disclosure discipline of `../gedi/EXPERIMENTS.md`
§0: the reproduction headline is never rewritten by the popoe line.

## Headline ledger

| # | Experiment | Archive number (source) | popoe entrypoint | Class | Reproduced | popoe commit / pod / date | Status |
|---|---|---|---|---|---|---|---|
| 1 | YCB-V full BOP AR | **0.7668** — `score_rules_ycbvg32m`; recipe: CNOS-FastSAM TOPK2 + gripper label pooling + grid-32 + O3D + fit×s_feat_1(×metric) | `examples/bop_eval.py --bop $BOP/ycbv --detections data/detections/cnos/cnos-fastsam_ycbv-test.json --merge ycbv --topk 2 --grid 32 --solver o3d --weights 1.0,0.7,0.5,0.3,0.2 --render-backend nvdiffrast --out … --cache … --cand-csv …` (full cmd → Run plan #1) | **GPU-POD** | — | — | ☐ |
| 2 | LM-O full BOP AR | **0.6726** — `lmog32`; CNOS∪SAM6D union detections + same pipeline | `examples/bop_eval.py --bop $BOP/lmo --sources cnos=…/cnos-fastsam_lmo-test.json,sam6d=…/sam6d_ism_lmo.json --merge none --topk 2 --grid 32 --solver o3d …` (full cmd → Run plan #2) | **GPU-POD** | — | — | ☐ |
| 3 | YCB-V AR(2/3) | 0.7528 (same run as #1) | same pose CSV as #1; score with gedi `scripts/freezev2_compute_ar_mssd_mspd.py` (LOCAL-CPU after #1) | **LOCAL-CPU** (post #1) | — | — | ☐ |
| 4 | LM-O AR(2/3) | 0.7324 (same run as #2) | same pose CSV as #2; same AR scorer as #3 | **LOCAL-CPU** (post #2) | — | — | ☐ |
| 5 | YCB-V grasp ADD(-S)@0.1d | **0.8173** (median 2.5 mm / 6.6°) — gedi `scripts/freezev2_grasp_eval.py`, locally recomputable from pose CSVs | **No popoe port** — run gedi script on #1 pose CSV: `BOP_PATH=$BOP/ycbv python ../gedi/scripts/freezev2_grasp_eval.py <pose.csv>` | **LOCAL-CPU** (post #1; GAP: no in-repo grasp CLI) | — | — | ☐ |
| 6 | LM-O grasp ADD(-S)@0.1d | **0.7617** (7.2 mm / 5.8°) | same as #5 with `BOP_PATH=$BOP/lmo` + #2 pose CSV | **LOCAL-CPU** (post #2; GAP: same) | — | — | ☐ |

## Contribution-level parity (secondary)

| Experiment | Archive result | popoe entrypoint | Class | Status |
|---|---|---|---|---|
| Adaptive visual weight | beats best-fixed on all 4 datasets | Built into `bop_eval.py --weights 1.0,0.7,0.5,0.3,0.2` (ChampionScorer per-target argmax over w). Cross-dataset 4-set claim still needs TUD-L / IC-BIN BOP data + GPU runs (not in this repo). Offline post-hoc: gedi `scripts/freezev2_adaptive_select.py` over per-w CSVs. | **GPU-POD** (YCB-V/LM-O covered by #1/#2); **GAP** for TUD-L/IC-BIN data | ☐ |
| Canonical-space scoring | 26-rule ablation; champion rule constant across datasets | Live rule = `ChampionScorer` (`s_icp * max(s_feat_1,0) * metric_fit?`). Offline re-sweep: `examples/rule_replay.py <cand.csv> --rule "s_icp*s_feat_1" --rule "s_icp*s_feat_1*metric_fit" --out-dir …` on a `--cand-csv` dump from #1/#2 (or existing popoe cands under `../gedi/ycbv_local_data/union_scoring_20260716/`). | **LOCAL-CPU** (once cand-csv exists) | ☐ |
| Gripper label pooling + metric_fit | obj20 +33.6 pt; 2×2 ablation | `bop_eval.py --merge ycbv` (pools 19:20, size_aware metric_fit) vs `--merge none` on YCB-V objs 19,20 (`--objs 19,20`). Live scorer: `ChampionScorer(size_aware=True)` for pooled pairs. | **GPU-POD** (subset ablation) | ☐ |
| Multi-mask / detection union (LM-O) | CNOS∪SAM6D +2.8 pt | Smoke (no GPU): `examples/union_smoke.py --dataset lmo --source sam6d=data/detections/sam6d/sam6d_ism_lmo.json`. Full AR: same as headline #2 (`--sources cnos=…,sam6d=…`). CNOS-only control: `--detections …/cnos-fastsam_lmo-test.json`. | **LOCAL-CPU** smoke + **GPU-POD** full | ☐ |

## Rules of engagement

1. **Fresh clone only.** Every pod run executes popoe from a `git clone` at a
   recorded commit — never hand-`scp`'d single files. (The gedi campaign lost
   time to stale bare-name module copies living only on the pod; that failure
   class ends here.)
2. Raw per-image CSVs from parity runs stay out of git (`data/` is ignored);
   this ledger records the number + commit + pod. Copy CSVs worth keeping to
   the gedi archive under `ycbv_local_data/` with a dated subdir.
3. One row per run: if a re-run disagrees with the archive beyond tolerance,
   do not overwrite — add a row and investigate before promoting either
   number.

## Gap list (archive capabilities popoe does not have yet)

Full gap report: `../gedi/EXPERIMENTS.md` Appendix B. Known candidates:

- Grasp-axis evaluation (ADD(-S)@0.1d) — currently only in gedi
  `scripts/freezev2_grasp_eval.py` (usable as external CLI on popoe pose CSVs;
  no `examples/` port yet).
- VSD computation / cross-check tooling (`freezev2_vsd_*.py`) — full BOP AR
  for #1/#2 still depends on gedi `scripts/freezev2_vsd_compute.py` after the
  pose CSV exists (AR(2/3) is local via `freezev2_compute_ar_mssd_mspd.py`).
- Adaptive visual-weight sweep harness (`freezev2_sweep_vis_weight.py`) —
  partially absorbed by `bop_eval --weights` + ChampionScorer; multi-dataset
  adaptive claim still needs TUD-L / IC-BIN.
- Full BOP RGB-D + CAD models for YCB-V / LM-O — **not** complete in the
  local archive (`../gedi/ycbv_local_data/bop_data/` is GT-meta + partial
  RGB for offline metrics). Full image trees live on the pod volume
  (`/workspace/bop_data/{ycbv,lmo}`). **Confirmed by Vincent 2026-07-22:
  the network volume (8rf4r42sf1) retains the full data tree and envs —
  mounting it is sufficient, no re-download needed. This gap is closed for
  pod runs.**

## Run plan (offline prep, 2026-07-22)

Path aliases used below (resolve on the machine you run on):

| Alias | Local (this workstation) | Pod (typical) |
|---|---|---|
| `$POPOE` | `/home/vincent/work/popoe` (fresh clone on pod) | `/workspace/popoe` |
| `$GEDI` | `/home/vincent/work/gedi` | `/workspace/gedi` or N/A |
| `$BOP` | **GAP** for full RGB-D — local has only `../gedi/ycbv_local_data/bop_data/` (GT + models_eval + sparse RGB) | `/workspace/bop_data` |
| `$DET` | `$POPOE/data/detections` (present locally; verified 2026-07-22) | copy from clone or volume |
| `$OUT` | N/A for GPU | `/workspace/results/parity_20260722` (create fresh) |

Env for GPU feature extraction (pod): `POPOE_GEDI_PATH`, `POPOE_BOP_TOOLKIT`,
CUDA + nvdiffrast (official numbers used `--render-backend nvdiffrast`). Fresh
`git clone` of popoe at a recorded commit — never scp single files.

Detection files verified present under `$POPOE/data/detections/`:

- `cnos/cnos-fastsam_ycbv-test.json`, `cnos/cnos-fastsam_lmo-test.json`
- `nids/nids_wa_sappe_ycbv.json`, `nids/nids_wa_sappe_lmo.json` (promotion line; not needed for gedi-headline #1/#2)
- `sam6d/sam6d_ism_ycbv.json`, `sam6d/sam6d_ism_lmo.json`, `sam6d/union_cnos_sam6d_lmo.reference.json`

### #1 — YCB-V full BOP AR 0.7668 (GPU-POD)

- **Class**: GPU-POD
- **Command** (from `$POPOE`, pod):

```bash
mkdir -p "$OUT"
uv run python examples/bop_eval.py \
  --bop "$BOP/ycbv" \
  --detections "$DET/cnos/cnos-fastsam_ycbv-test.json" \
  --merge ycbv \
  --topk 2 \
  --grid 32 \
  --solver o3d \
  --weights 1.0,0.7,0.5,0.3,0.2 \
  --render-backend nvdiffrast \
  --out "$OUT/parity_ycbv_g32m.csv" \
  --cache "$OUT/cache_ycbv_g32m" \
  --cand-csv "$OUT/parity_ycbv_g32m_cands.csv"
```

- **Data deps**: full YCB-V BOP test RGB-D + `models/` + `test_targets_bop19.json`
  (**pod only**); CNOS-FastSAM JSON (local OK under `$DET/cnos/`).
- **Post (LOCAL-CPU or pod CPU)**: AR(2/3) + VSD → full BOP AR:

```bash
BOP_PATH="$BOP/ycbv" python "$GEDI/scripts/freezev2_compute_ar_mssd_mspd.py" \
  "$OUT/parity_ycbv_g32m.csv"
# VSD (needs models + depth; typically pod):
python "$GEDI/scripts/freezev2_vsd_compute.py" "$OUT/parity_ycbv_g32m.csv"
```

- **Preconditions**: nvdiffrast on matching GPU arch (4090 sm_89); GeDi + DINOv2
  loadable; resume-safe if `--out` partially written.
- **Est. GPU wall**: ~15–22 h on RTX 4090 for full 21-obj set (formal union2
  fullset wall was ~22 h; CNOS-only is lighter on candidates, still O(10+ h)).
  Optional 2-way split: `--objs 1,2,…,10` / `11,…,21` then merge CSVs.

### #2 — LM-O full BOP AR 0.6726 (GPU-POD)

- **Class**: GPU-POD
- **Command**:

```bash
uv run python examples/bop_eval.py \
  --bop "$BOP/lmo" \
  --sources "cnos=$DET/cnos/cnos-fastsam_lmo-test.json,sam6d=$DET/sam6d/sam6d_ism_lmo.json" \
  --merge none \
  --topk 2 \
  --grid 32 \
  --solver o3d \
  --weights 1.0,0.7,0.5,0.3,0.2 \
  --render-backend nvdiffrast \
  --out "$OUT/parity_lmo_g32_union.csv" \
  --cache "$OUT/cache_lmo_g32" \
  --cand-csv "$OUT/parity_lmo_g32_union_cands.csv"
```

- **Data deps**: full LM-O BOP test (**pod**); CNOS + SAM6D JSON (local OK).
- **Post**: same AR/VSD scripts with `BOP_PATH=$BOP/lmo`.
- **Preconditions**: same as #1. Do **not** pass `--use-s-coarse` (hurts LM-O;
  that is the popoe promotion line, not the gedi headline).
- **Est. GPU wall**: ~3–5 h on 4090 (union3 L40S log ~2.7 h for related run).

### #3 / #4 — AR(2/3) (LOCAL-CPU after #1 / #2)

- **Class**: LOCAL-CPU (once pose CSVs exist)
- **Command**: see Post blocks under #1 / #2 (`freezev2_compute_ar_mssd_mspd.py`).
- **Data deps**: pose CSV + `$BOP/{ycbv,lmo}` GT (`scene_gt.json`,
  `models_eval/`). Local archive GT path usable:
  `../gedi/ycbv_local_data/bop_data/{ycbv,lmo}/` (verified present).
- **Est.**: minutes on CPU; no GPU.

### #5 / #6 — Grasp ADD(-S)@0.1d (LOCAL-CPU after #1 / #2; GAP port)

- **Class**: LOCAL-CPU post-pose; **GAP** = no popoe-native grasp CLI
- **Command** (gedi script; path hardcoded default `/workspace/bop_toolkit`
  — set `PYTHONPATH` / edit sys.path or run where toolkit lives):

```bash
# YCB-V (#5) — target archive 0.8173
BOP_PATH="$BOP/ycbv" python "$GEDI/scripts/freezev2_grasp_eval.py" \
  "$OUT/parity_ycbv_g32m.csv"

# LM-O (#6) — target archive 0.7617
BOP_PATH="$BOP/lmo" python "$GEDI/scripts/freezev2_grasp_eval.py" \
  "$OUT/parity_lmo_g32_union.csv"
```

- **Data deps**: pose CSV + `models_eval/*.ply` + per-scene `scene_gt.json`
  (local `../gedi/ycbv_local_data/bop_data/` sufficient for metrics if scenes
  in the CSV are covered).
- **Est.**: <5 min CPU each; zero GPU.
- **Sanity without new poses**: recompute on existing gedi champion CSV under
  `../gedi/ycbv_local_data/freezev2/score_rules_ycbvg32m/rule_champion_size.csv`
  (archive path for 0.7668 / grasp 0.8173 chain).

### C1 — Adaptive visual weight (GPU-POD partial / GAP multi-dataset)

- **Class**: GPU-POD for YCB-V/LM-O (already inside #1/#2 via `--weights`);
  **GAP** for TUD-L + IC-BIN (no data in this workspace).
- **Command**: no extra live flag — ChampionScorer selects best w per target.
  Offline histogram over fixed-w CSVs (gedi):
  `python "$GEDI/scripts/freezev2_adaptive_select.py" out.csv w1.csv w0.7.csv …`
- **Est. GPU**: covered by #1/#2 wall time.

### C2 — Canonical-space scoring / 26-rule replay (LOCAL-CPU)

- **Class**: LOCAL-CPU
- **Command** (popoe column names: `s_icp`, `s_feat_1`, `metric_fit`, optional
  `s_coarse` — **not** the older gedi `icp_fit` header):

```bash
uv run python examples/rule_replay.py \
  ../gedi/ycbv_local_data/union_scoring_20260716/popoe_ycbv_formal_A_cands.csv \
  --rule "s_icp*s_feat_1" \
  --rule "s_icp*s_feat_1*metric_fit" \
  --rule "s_icp*s_feat_1*metric_fit*s_coarse" \
  --out-dir /tmp/popoe_rule_replay_ycbv
```

- **Data deps**: any popoe `--cand-csv` dump (existing union_scoring cands
  verified under `../gedi/ycbv_local_data/union_scoring_20260716/`).
- **Est.**: <1 min CPU. Full 26-rule grid is the same tool with more `--rule`s.

### C3 — Gripper pooling 2×2 (GPU-POD subset)

- **Class**: GPU-POD
- **Command** (minimal ablation on clamps only):

```bash
# pool + metric_fit (default merge=ycbv → size_aware on 19/20)
uv run python examples/bop_eval.py \
  --bop "$BOP/ycbv" \
  --detections "$DET/cnos/cnos-fastsam_ycbv-test.json" \
  --objs 19,20 --merge ycbv --topk 2 --grid 32 --solver o3d \
  --render-backend nvdiffrast \
  --out "$OUT/ab_clamp_merge.csv" --cache "$OUT/cache_clamp"

# no pool control
uv run python examples/bop_eval.py \
  --bop "$BOP/ycbv" \
  --detections "$DET/cnos/cnos-fastsam_ycbv-test.json" \
  --objs 19,20 --merge none --topk 2 --grid 32 --solver o3d \
  --render-backend nvdiffrast \
  --out "$OUT/ab_clamp_nopool.csv" --cache "$OUT/cache_clamp"
```

- **Est. GPU**: ~1–2 h (300 targets × 2 configs) on 4090.

### C4 — Multi-mask union LM-O (LOCAL-CPU smoke + GPU-POD full)

- **Class**: LOCAL-CPU smoke; GPU-POD full (= #2)
- **Smoke**:

```bash
uv run python examples/union_smoke.py --dataset lmo \
  --source sam6d=data/detections/sam6d/sam6d_ism_lmo.json
uv run python examples/union_smoke.py --dataset ycbv \
  --source sam6d=data/detections/sam6d/sam6d_ism_ycbv.json
```

- **Full AR**: command under #2; CNOS-only control uses `--detections` single file.
- **Est. GPU**: same as #2.

### Pod session budget (all GPU-POD items in one session)

| Item | Est. GPU h (4090) |
|---|---|
| #1 YCB-V full parity | 15–22 |
| #2 LM-O full parity | 3–5 |
| C3 clamp 2×2 (optional same session) | 1–2 |
| VSD post (#1+#2) | ~0.3–0.5 |
| **Total** | **~20–30 h** |
| **Cost @ $0.69/hr** | **~$14–21** |

LOCAL-CPU items (#3–#6, C2, C4 smoke, pytest) add negligible $ and can run
on this workstation after CSVs land (or on the pod after GPU finishes).

## Offline verification log

Prep date: **2026-07-22**. Host: local workstation (no NVIDIA driver —
`nvidia-smi` failed; no GPU smoke of feature stack). Scope: path existence,
CLI flags from code, light CPU tests. GPU parity numbers still ☐.

### Path / artifact checks (read-only)

| Path | Role | Present? |
|---|---|---|
| `data/detections/cnos/cnos-fastsam_{ycbv,lmo}-test.json` | #1/#2 detections | yes (ls 2026-07-22) |
| `data/detections/sam6d/sam6d_ism_{ycbv,lmo}.json` | #2 union | yes |
| `data/detections/nids/nids_wa_sappe_{ycbv,lmo}.json` | promotion line only | yes |
| `../gedi/ycbv_local_data/freezev2/` | gedi g32 candidates + score_rules | yes |
| `../gedi/ycbv_local_data/union_scoring_20260716/` | popoe formal CSVs + cands + grasp logs | yes |
| `../gedi/ycbv_local_data/bop_data/{ycbv,lmo}/` | GT meta + models_eval (+ sparse RGB) | yes — **not** full BOP RGB-D |
| `../gedi/scripts/freezev2_grasp_eval.py` | #5/#6 external CLI | yes |
| `../gedi/scripts/freezev2_compute_ar_mssd_mspd.py` | #3/#4 AR(2/3) | yes |
| `../gedi/scripts/freezev2_vsd_compute.py` | full BOP AR VSD leg | yes |
| `/workspace/bop_data/{ycbv,lmo}` | full RGB-D for GPU | **not on this host** (pod volume) |
| CUDA / nvdiffrast | feature extraction | **unavailable locally** |

### CLI flags verified from code (not memory)

- `examples/bop_eval.py`: mutually exclusive `--detections` / `--sources`;
  defaults `--topk 2`, `--grid 32`, `--solver o3d`, `--merge ycbv`,
  `--weights` = recipes.WEIGHTS `(1.0,0.7,0.5,0.3,0.2)`,
  `--render-backend nvdiffrast|trimesh|auto`; Champion rule via
  `ChampionScorer` / `stages_for_object` (see `src/popoe/scoring.py`,
  `src/popoe/freeze/recipes.py`).
- `examples/rule_replay.py`: product rules over cand-csv columns; zero GPU.
- `examples/union_smoke.py`: defaults CNOS+NIDS under `data/detections/`;
  `--source name=path` overrides/adds; no RGB-D required.
- `examples/pipeline_selfcheck.py` / `solver_swap_demo.py`: need CUDA + full
  BOP mesh/RGB — **not run** locally (GPU-POD / >10 min risk).

### Smoke commands (results filled after run)

| Command | Est. | Result (review run, 2026-07-22) |
|---|---|---|
| `uv run pytest tests/` | <5 min | **120 passed**, 12.5 s |
| `uv run python examples/union_smoke.py --dataset ycbv --source sam6d=data/detections/sam6d/sam6d_ism_ycbv.json` | <2 min | **OK** end-to-end (3-way union, 746 champions) |
| `uv run python examples/union_smoke.py --dataset lmo --source sam6d=data/detections/sam6d/sam6d_ism_lmo.json` | <2 min | **OK** end-to-end (393 champions) |
| `uv run python examples/rule_replay.py …/popoe_ycbv_formal_A_cands.csv --rule "s_icp*s_feat_1" --rule "s_icp*s_feat_1*metric_fit" --rule "s_icp*s_feat_1*metric_fit*s_coarse" --out-dir …` | <1 min | **OK** — 21 800 hyps / 1 669 targets; ×metric_fit flips 44.0% vs formal baseline; +s_coarse flips 0.2% (formal baseline is itself s_coarse-arbitrated — consistency check ✓). Original plan referenced `popoe_ycbv_union2_cands.csv` which does not exist; corrected to `popoe_ycbv_formal_A_cands.csv`. |
| `uv run python examples/pipeline_selfcheck.py …` | needs GPU | **skipped** (no local GPU) |
| full `bop_eval` parity | 15–22 h GPU | **not run** (zero-GPU prep) |
