# Adversarial review, path B2: five new datasets × four official detection sources (2026-08-06)

Scope: pre-freeze hunt for where the pipeline's assumptions break on the five
BOP sets it has never formally run (tless, tudl, icbin, itodd, hb) and on the
new official detection files (CNOS / SAM6D-441 / NIDS WA_Sappe / MUSE).
Method: code read of the ingestion → selection → layout → time chain, plus
decode/coverage/schema checks against the REAL payloads of all 20 new files
(4 sources × 5 sets) and the 8 lmo/ycbv files. No src edits.

Severity legend: **blocks-pin** = resolve (fix or explicit decision) before the
18-run table is frozen; **must-disclose** = can run as-is but the deviation
must be in the run table / thesis; **minor** = note, no action forced.

---

## Findings

### F1 · blocks-pin — cross-source duplicate instances eat `inst_count` slots (E-4way × icbin/tless/itodd/hb)

- `src/popoe/segmentor_detections.py:396-411` — the union deliberately does
  NOT dedupe across sources ("FreeZe's top-M union without filtering");
  `iou_dedupe` is scoped per source (`kept_by_source`).
- `src/popoe/adapters.py:134-146` — `select_top_instances` treats every
  detection index as a candidate DISTINCT instance and keeps the top
  `inst_count` champions by score, with no spatial exclusivity.

Both design decisions are individually documented and correct on
`inst_count==1` (all of lmo/ycbv, i.e. every formal run so far): the scorer
arbitrates and one winner survives. They compose badly on `inst_count>1`:
the same physical instance proposed by 2–4 sources yields 2–4 near-identical
high-scoring champions that occupy 2–4 of the k slots; genuinely distinct
instances ranked below the duplicates are pushed out of the row budget.
Recall on crowded targets is structurally capped at roughly
`inst_count / n_sources_that_saw_it`, independent of pose quality.

Concrete: icbin obj 2 has `inst_count` up to 19; the E-4way arm feeds up to
4 × 19 = 76 candidates per target and writes 19 rows chosen purely by score.
tless (multi-copy scenes) and itodd (bin scenes) hit the same path. BOP's
greedy matcher counts each duplicate at most once, so every duplicated slot
is a lost instance.

Resolution options before pinning:
1. selection-time duplicate suppression when `inst_count > 1` (greedy IoU
   over champion masks before truncating to k) — smallest change with clear
   semantics; or
2. run E-4way as-is and pre-register the cap as a known property of the
   unfiltered union on multi-instance sets (then E-cnos, single-source, is
   the comparable multi-instance line).
Silently freezing without choosing is the only wrong option: the number the
union arm produces on icbin/tless/itodd is otherwise not interpretable as
"union helps/hurts".

Note the single-source version of the same issue (one detector's overlapping
proposals at IoU < 0.9 both kept) is much milder — detectors ship their own
NMS, and this matches FreeZe's own single-source semantics — must-disclose at
most.

### F2 · blocks-pin (operational) — the frozen table's detection paths dangle on any other host

- `.gitignore:17` (`data/detections/**`) and `.gitignore:32` (`outputs/`):
  none of the 28 detection inputs are in git.
- 19 of them are ABSOLUTE-path symlinks into
  `outputs/seg_ap_20260725T223014Z/official_submissions/` (all sam6d_official,
  all muse, 5 of 7 nids). On the pod those resolve to
  `/home/vincent/work/popoe/outputs/...` → FileNotFoundError at segmentor
  init (loud, at least — but it kills the run at t=0 after pod setup cost).

The freeze must include a materialization step: `cp -L` (or B2 restore of the
real files) into `data/detections/` on the run host, then SHA256 against the
tables in `data/detections/*/PROVENANCE.md` (EXPERIMENT_PLAN already requires
checksums for the 28 files). A frozen run table that names symlink paths
without this step is not executable anywhere but this laptop.

### F3 · must-disclose — top-K floor is dataset-wide max `inst_count`, not per-target N+1 (A2, now quantified)

- `examples/bop_eval.py:364-370` (`floored_topk`), `:729-737` — the floor is
  `max(--topk, max inst_count over the WHOLE targets file)` and applies to
  every (source, label) bucket of every image.

On icbin (max inst 19) every bucket keeps up to 19 masks even for
single-instance targets. Measured from the actual files (bucket counts,
assumed floors icbin 19 / tless 4 / itodd 6 / hb 2 — exact values need the
targets files, absent locally, see F6):

| set | masks kept, per source (cnos/sam6d/nids/muse) | union total |
|---|---|---|
| icbin | 4933 / 4961 / 3796 / 4517 | ≈ 18.2k |
| tless | 32794 / 35687 / 18354 / 24488 | ≈ 111k (upper bound: counts include non-target labels) |
| itodd | 9955 / 16463 / 5170 / 5881 | ≈ 37k |
| hb | 5720 / 7054 / 4875 / 5102 | ≈ 22.8k |

Each kept mask costs one GeDi+DINO target encode plus 5 weights ×
solve/refine/score. Two consequences: (a) E-4way tless/icbin/itodd runtime is
dominated by the floor, budget accordingly (tless union is O(100k) encodes);
(b) the "M" of any M=2N claim is NOT the paper's per-target N+1 —
REPRODUCTION.md:92 already blocks that claim; this stays true on all five new
sets and is most visible on icbin.

### F4 · must-disclose — itodd and hb have NO local score signal; server-only arms

itodd and hb test GT is withheld (BOP evaluates server-side). Consequences:
- `--probe-corr` crashes there (`examples/bop_eval.py:1111-1118` reads
  `scene_gt.json`); diagnostic only, don't schedule it on those sets.
- local AR (`src/popoe/metrics/ar.py`) cannot gate those 4 of the 18 runs
  (E-cnos/E-4way × itodd/hb). The only acceptance signal is a BOP submission
  round-trip. The campaign schedule must not put itodd/hb runs last-minute
  before a deadline, and the freeze should say which sets are locally gated
  (tless/tudl/icbin: public test GT) vs server-gated (itodd/hb).

### F5 · must-disclose — visual branch is off-distribution on tless and itodd

- tless: `models_cad` (layout default, `src/popoe/datasets/bop.py:38-39`) is
  colourless CAD → grey query renders; scene photos are RGB primesense.
  Mitigated by tless objects being largely uniform grey plastic, but the
  DINO half and the weight sweep (tuned on lmo/ycbv) were never validated on
  a colourless-CAD set.
- itodd: grayscale photos replicated to 3 channels
  (`examples/bop_eval.py:334-361`) meet colourless-CAD renders — both sides
  grey, self-consistent but untested; `--render-rerank` (the 905 recipe's
  third component) scores DINO patch cosine on these — untested territory.
- The uint8 guard at `examples/bop_eval.py:353-358` hard-fails on 16-bit
  gray tifs. Loud, correct — but PREFLIGHT one itodd image before committing
  GPU time, since the failure would otherwise land mid-campaign.

### F6 · must-disclose — the five new sets' data trees are not on this machine; floors and layout never touched disk

`bop_data/` holds lmo + ycbv only. Everything layout-dependent for the new
five (split dirs `test_primesense` for tless/hb, `gray/*.tif` for itodd,
`models_cad` + its `models_info.json` for tless `--tau-diameter`, targets
files with real `inst_count`) is encoded in `BOP_LAYOUTS`
(`src/popoe/datasets/bop.py:36-50`, matches bop_toolkit dataset_params) but
has never been exercised against real directories. The design fails loud
(unknown dataset fatal, missing image fatal by default,
`--allow-missing-images` opt-in at `examples/bop_eval.py:459-465,1049-1070`),
so the risk is wasted pod hours, not silent zeros. Freeze should attach a
preflight: for each set, load 1 image + 1 mesh + targets file and print the
`dataset=... split=... images=... models=...` line (`bop_eval.py:647-649`)
before scheduling the real run.

### F7 · minor — stale `--merge ycbv` command templates would poison tless/itodd/hb

`resolve_merge` 'auto' (`examples/bop_eval.py:284-301`) correctly pools only
on ycbv. But the literal `--merge ycbv` is still accepted on any dataset and
would pool unrelated objs 19/20 AND hand them the size-aware scorer
(`stages_for_object(size_aware=obj_id in merge)`, `bop_eval.py:899`). Risk is
purely copy-pasted old commands. The frozen run table must spell `--merge
auto` (or `none`) per run; `--dual-assign` already refuses without a merge.

### F8 · minor (verified good) — time normalization works for the 4-JSON union

Checked all 28 files: every record carries a float `time`, constant within
each image (0 varying records anywhere), and every target image of every set
is covered by ALL four sources (missing_vs_union = 0 across the board). So
`examples/bop_time_normalize.py` `--detections` × 4 (`:81-97`; max-per-image
is exact under constancy, sum across files at `:100-118`) produces exact BOP
per-image totals with `uncovered == 0`. Disclose the standard caveat:
detection `time` is the publishers' hardware, pose time is ours. Do not
re-run the normalizer on its own output (guarded, `:26-30,67-71`).

### F9 · minor (verified good) — ingestion assumptions hold against the real payloads

Sampled/decoded from every one of the 20 new files
(`load_bop_detections` + `decode_detection_mask`):
- schema: flat lists; int scene_id/image_id/category_id; float score;
  `segmentation` is always an RLE dict with list `counts` (uncompressed) —
  no polygons, no compressed-string counts, no stringified fields in ANY of
  the five new sets' files (the stringified-NIDS trap exists only in the
  in-tree lmo/ycbv Box files, which the loader handles: 
  `src/popoe/segmentor_detections.py:50-112`).
- sizes match the datasets: tless (540,720), tudl/icbin/hb (480,640), itodd
  (960,1280); uniform within each file → no mask/image shape mismatch.
- coverage: image sets identical across the four sources per set (tless
  1000, tudl 600, icbin 150, itodd 721 (scene 1 only), hb 300 (scenes
  3/5/13 = primesense bop19)); category_id exactly 1..n_objs (30/3/2/28/33)
  — direct obj_id semantics, no COCO remap anywhere.
- decode integrity: 40 random masks per file decoded and cross-checked
  against the record's own bbox — no transpose, no corruption, no empty
  masks; worst mask-extent-vs-bbox IoU 0.40 (loose boxes on fragmented
  masks, normal).
- scores all in [0.087, 0.892] — no >1/negative surprises; `min_pixels=100`
  drops at most 1 record per file (3 files affected) → no silent small-object
  starvation, including itodd.
- `infer_detection_source` (`examples/bop_eval.py:139-152`) tags every new
  filename correctly (checked against the actual paths; `muse-repro` guard
  ordering is right).
- empty-detection images: `segment()` returns `[]` → exactly `inst_count`
  zero rows, resume invariant holds (`bop_eval.py:1167-1193`,
  `adapters.py:102-131`).
- source-union name collisions rejected (`segmentor_detections.py:297-302`).

This discharges the ingestion half of TRIAGE_20260806 A6 (files never ran
the pipeline): decode and coverage are now verified offline; what remains
untested is only the GPU stages downstream of the mask, which F6's preflight
plus the first scheduled run covers.

### F10 · minor (verified good) — MSPD 640/W rescale is handled

`src/popoe/metrics/aggregate.py:62-94` and `src/popoe/metrics/ar.py:52-59,
138-142`: per-scene width normalization with correct defaults (tless 720,
itodd 1280). No hardcoded-640 trap for the new sets in the local metric
path; server evaluation is authoritative anyway.

---

## Summary disposition

| # | Finding | Severity |
|---|---|---|
| F1 | union duplicates vs inst_count>1 selection | blocks-pin (decision required) |
| F2 | gitignored + absolute-symlink detection inputs | blocks-pin (ops step in freeze) |
| F3 | dataset-max top-K floor; compute + M semantics | must-disclose (A2 quantified) |
| F4 | itodd/hb server-only gating | must-disclose (schedule) |
| F5 | tless/itodd visual-branch domain | must-disclose |
| F6 | five sets' trees unpreflighted | must-disclose (preflight step) |
| F7 | literal `--merge ycbv` on other sets | minor (template hygiene) |
| F8 | 4-JSON time normalization | minor, verified good |
| F9 | ingestion vs real payloads | minor, verified good |
| F10 | MSPD rescale | minor, verified good |
