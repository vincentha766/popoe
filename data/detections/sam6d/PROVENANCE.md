# Official SAM6D-FastSAM(RGB) detections (source tag: `sam6d` official line)

BOP `method_info/546`, task "Model-based 2D segmentation of unseen objects".
Downloaded 2026-08-06 (LM-O/YCB-V 2026-07-26 as public reference copies).
All seven from the **08:23–08:26 segmentation batch** (2024-03-22).
⚠️ The same-day **08:37–08:38 batch 8018–8024 is a DIFFERENT TASK** —
2D detection, bbox only, no `segmentation` field. Never mix.

⚠️ **Two SAM6D provenances live in this directory — do not conflate:**
- `sam6d_ism_{lmo,ycbv}.json` — **our local ISM runs** (official code, our
  execution); used by the frozen Phase D `tuned-4way`/`faithful-3way` recipes.
  NOT byte-identical to the official submissions.
- `sam6d_official_<ds>.json` (symlinks below) — the **authors' official BOP
  submissions**; per Vincent 2026-08-06, the Phase E seven-set lines use these.

| Dataset | BOP submission | Records | SHA256 |
|---|---|---|---|
| LM-O | [8005](https://bop.felk.cvut.cz/sub_info/8005/) | 12950 | `31fe66fe4ae9772b37d30fcbeb322186ddafb9402d2981882504c7b79cd7f73b` |
| TUD-L | [8003](https://bop.felk.cvut.cz/sub_info/8003/) | 16353 | `42b94fd25a1f8ccfb1be855e04f161679b8932f99779e29d67ffc5f41ca1ebfe` |
| T-LESS | [8004](https://bop.felk.cvut.cz/sub_info/8004/) | 56942 | `ca66acdce2ccc13eb1d2b92f09e92bdf0a627f28f8daa2698bbe61628d161918` |
| IC-BIN | [8006](https://bop.felk.cvut.cz/sub_info/8006/) | 6166 | `ffa2c2fd0ea91b78f0e88453d7e74d9aa600d177a16481a0d50e72517948b093` |
| ITODD | [8007](https://bop.felk.cvut.cz/sub_info/8007/) | 10625 | `c058878ac377799f7483797ea2dfe1e1dd90bebc69eba27f55ae82ad926716a7` |
| HB | [8008](https://bop.felk.cvut.cz/sub_info/8008/) | 13240 | `0b7fa39669bb9c7909930f8bd37a0bbef648b47d1bb43156d5452c79bb3feafc` |
| YCB-V | [8009](https://bop.felk.cvut.cz/sub_info/8009/) | 30374 | `64f50fbbe61454ef99881ba09c060df3f1baf00589a751c21327cdd202513a13` |

Files live in `outputs/seg_ap_20260725T223014Z/official_submissions/`;
symlinked here as `sam6d_official_<ds>.json`. Schema per record:
`scene_id, image_id, category_id, bbox, score, time, segmentation` (verified
2026-08-06; an earlier truncated HB download was caught by JSON parse-check —
always parse after download, size alone lies).
