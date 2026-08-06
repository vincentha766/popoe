# Official SAM6D detections (source tag: `sam6d` official line)

**E-4way source = method 441 "SAM6D"** (Vincent 2026-08-06, amended same day
from 546 before any server score): the strongest official SAM6D variant
(mean seg AP 0.481; family spread 441/545/466/546 = 5.3pt). Seg batch
**6965-6971** (2023-12-05 07:46-07:50). ⚠️ 441's method page mixes THREE
tasks — 6951-6957 is 2D detection (boxes), 7088-7094 and 7238-7244 are 6D
localization (poses). The seg batch was identified by matching per-set AP
to the leaderboard row (LM-O 0.460). Always check the Task field.

⚠️ **Three SAM6D provenances now exist — never conflate:**
- `sam6d_ism_{lmo,ycbv}.json` — our local ISM runs (FastSAM proposals);
  frozen Phase D recipes only. NOT byte-identical to any official file.
- `sam6d_official_<ds>.json` (symlinks) — **now point to method 441 files**
  (`sam6d-sam441_*-test_official.json`); the Phase E E-4way inputs.
- 546 FastSAM(RGB) files (`sam6d-fastsamrgb_*-test_official.json`) — kept in
  the archive dir as the public reference matching our ISM lineage; no longer
  wired into `data/detections/`.

## Method 441 seg batch (E-4way inputs)

| Dataset | BOP submission | Records | SHA256 |
|---|---|---|---|
| LM-O | [6967](https://bop.felk.cvut.cz/sub_info/6967/) | 21264 | `638a933c0f3f404086f975050524ead00b23f6c081d77a1dce99443fab781108` |
| TUD-L | [6965](https://bop.felk.cvut.cz/sub_info/6965/) | 29482 | `267784437d15d97061dc30248bacdb08631780385fc7b86507818fd7ef63a6ab` |
| T-LESS | [6966](https://bop.felk.cvut.cz/sub_info/6966/) | 61082 | `e63e91376d3c116ea39aec2b5c173b0358f099fbb260275ab69fe48200e6fdf6` |
| IC-BIN | [6968](https://bop.felk.cvut.cz/sub_info/6968/) | 7778 | `3e4797bfda1dc2ca7514018ed082b6c8678351bf6905f2803725825bc167cef8` |
| ITODD | [6969](https://bop.felk.cvut.cz/sub_info/6969/) | 18915 | `d0511f138d0e509ee3fb028e5d3c438fa1f2cb6ae27e1ef4fc6d22b3968595e2` |
| HB | [6971](https://bop.felk.cvut.cz/sub_info/6971/) | 20142 | `f22f496109341f8bb0f03c0d33476bb0af69f468f604afbd7fc03c898dc2d39a` |
| YCB-V | [6970](https://bop.felk.cvut.cz/sub_info/6970/) | 46382 | `2288e24bfcbed29aedb719b53bff40f1a558a47f02392d5da0b6dabb2539abf8` |

## Method 546 FastSAM(RGB) seg batch (reference only, 8003-8009)

| Dataset | BOP submission | Records | SHA256 |
|---|---|---|---|
| LM-O | [8005](https://bop.felk.cvut.cz/sub_info/8005/) | 12950 | `31fe66fe4ae9772b37d30fcbeb322186ddafb9402d2981882504c7b79cd7f73b` |
| TUD-L | [8003](https://bop.felk.cvut.cz/sub_info/8003/) | 16353 | `42b94fd25a1f8ccfb1be855e04f161679b8932f99779e29d67ffc5f41ca1ebfe` |
| T-LESS | [8004](https://bop.felk.cvut.cz/sub_info/8004/) | 56942 | `ca66acdce2ccc13eb1d2b92f09e92bdf0a627f28f8daa2698bbe61628d161918` |
| IC-BIN | [8006](https://bop.felk.cvut.cz/sub_info/8006/) | 6166 | `ffa2c2fd0ea91b78f0e88453d7e74d9aa600d177a16481a0d50e72517948b093` |
| ITODD | [8007](https://bop.felk.cvut.cz/sub_info/8007/) | 10625 | `c058878ac377799f7483797ea2dfe1e1dd90bebc69eba27f55ae82ad926716a7` |
| HB | [8008](https://bop.felk.cvut.cz/sub_info/8008/) | 13240 | `0b7fa39669bb9c7909930f8bd37a0bbef648b47d1bb43156d5452c79bb3feafc` |
| YCB-V | [8009](https://bop.felk.cvut.cz/sub_info/8009/) | 30374 | `64f50fbbe61454ef99881ba09c060df3f1baf00589a751c21327cdd202513a13` |

## Local ISM runs (frozen Phase D lineage, superseded — no arm uses these)

| File | SHA256 |
|---|---|
| `sam6d_ism_lmo.json` | `19f44ba740e422d3b7ad09d08656bcca03092a4dc1e21707d5f35243e49f1107` |
| `sam6d_ism_ycbv.json` | `dcadea8f62d37779747c52e153180c36d72b37fde3a3cd08ba36d02d75ef081c` |

## Local test fixture (not a detection input)

| File | SHA256 |
|---|---|
| `union_cnos_sam6d_lmo.reference.json` (union-ingestion parity reference, `tests/test_union_reference_xval.py`) | `5c11cf2d5d98db241798f55976c7cdcff9561350cd87cc46040747e599a7c40b` |

Files live in `outputs/seg_ap_20260725T223014Z/official_submissions/`;
symlinked here as `sam6d_official_<ds>.json`. Schema per record:
`scene_id, image_id, category_id, bbox, score, time, segmentation` (verified
2026-08-06; an earlier truncated HB download was caught by JSON parse-check —
always parse after download, size alone lies).
