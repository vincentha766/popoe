# Official MUSE detections (source tag: `muse`)

BOP `method_info/873`, task "Model-based 2D segmentation of unseen objects".
**Nothing in popoe writes this name** — popoe's own reimplementation writes
`source="muse-repro"` and stays a separate path.

All seven BOP-Classic-Core mask files are from the same authored batch
(2025-08-26 05:14–05:16 UTC). MUSE has no public code, but every core-set
mask file is publicly downloadable from the `sub_info` page ("Download
submission"). Save as `muse-full_<ds>-test.json` in this directory.

| Dataset | BOP submission | Batch (UTC) | Records | Scenes | Objs | SHA256 |
|---|---|---|---:|---:|---:|---|
| LM-O | [29108](https://bop.felk.cvut.cz/sub_info/29108/) | 2025-08-26 05:14 | 7146 | 1 | 8 | `55061983089d6236c19cb9b6a8a6c754388d146287be45ec40ceb9c32dbe3003` |
| IC-BIN | [29109](https://bop.felk.cvut.cz/sub_info/29109/) | 2025-08-26 05:14 | 4731 | 3 | 2 | `34a2a40b3c716bb3c36b0739d49ebc885019cb6d97079d4e5e1ba9c743ed1427` |
| TUD-L | [29110](https://bop.felk.cvut.cz/sub_info/29110/) | 2025-08-26 05:15 | 15736 | 3 | 3 | `38dc40cfa75f22a74f1f85cb10fb2283adb99db65ea496dff68bf216beeccb8b` |
| T-LESS | [29111](https://bop.felk.cvut.cz/sub_info/29111/) | 2025-08-26 05:15 | 26511 | 20 | 30 | `78bcdab72d0eac44ab5b8477eec9e229fdaa2e61fdc69bcec48be46a3f230482` |
| ITODD | [29112](https://bop.felk.cvut.cz/sub_info/29112/) | 2025-08-26 05:15 | 6320 | 1 | 28 | `2d34ebce3a464f129f6cdc8770686df56869eafa8bd8fff135fbfecb3c65813a` |
| HB | [29063](https://bop.felk.cvut.cz/sub_info/29063/) | 2025-08-26 05:15 | 6440 | 3 | 33 | `c0e0802a3db1e2394507099098ed5000208d93e1701f8b19850d6cd6d7d59d1d` |
| YCB-V | [29113](https://bop.felk.cvut.cz/sub_info/29113/) | 2025-08-26 05:16 | 16902 | 12 | 21 | `b4703a218d13f707d47556b2733eeddc38fea7d89bf927d113da25349c74f497` |

Every record carries `scene_id, image_id, category_id, bbox, score, time,
segmentation`. Download URL pattern: the `sub_info` page carries a
"Download submission" link to `/media/subs/muse_<dataset>-test_<uuid>.json`.
The UUIDs are not derivable — follow the page, then
`python scripts/freeze_detections.py --check`.

## Two traps if these are ever re-fetched

1. **HB is `29063`**, outside the otherwise contiguous 29104–29124 block. Do not
   infer submission IDs by counting.
2. **The 05:47–05:48 submissions are a DIFFERENT TASK, not a re-run.**
   `29115`–`29121` cover the same seven sets but their task is
   **"Model-based 2D detection of unseen objects"** — bbox only, **no
   `segmentation` field at all** (verified 2026-08-06: every record has exactly
   `bbox, category_id, image_id, scene_id, score, time`). They are useless as
   FreeZe mask input. All files here are from the 05:14–05:16
   **segmentation** batch, which is the only one that carries masks.

   Their AP is therefore **box AP, not mask AP** and must never be compared
   against the segmentation rows above:

   | Dataset | seg AP (used here) | det AP (05:47–05:48, boxes only) |
   |---|---:|---:|
   | LM-O | 0.477 (`29108`) | 0.511 (`29115`) |
   | IC-BIN | 0.433 (`29109`) | 0.337 (`29118`) |
   | TUD-L | 0.573 (`29110`) | 0.601 (`29117`) |
   | T-LESS | 0.478 (`29111`) | 0.494 (`29116`) |
   | ITODD | 0.391 (`29112`) | 0.490 (`29119`) |
   | HB | 0.635 (`29063`) | 0.616 (`29121`) |
   | YCB-V | 0.690 (`29113`) | 0.684 (`29120`) |

   LM-O det/seg carry identical `bbox`, `score`, `category_id` and record
   count (7146) — the detection file is the segmentation file with masks
   stripped. Do not put the 05:47–05:48 files in this directory.

Also on 873 but outside BOP-Classic-Core (not fetched): IPD `29104`,
XYZ-IBD `29114`, HOPEv2 `29122`, HOT3D `29123`, HANDAL `29124`.

## Which AP number to quote

The **public row** AP is what appears on the `sub_info` page (LM-O 0.477,
YCB-V 0.690). Local re-evaluation of the same file differs by evaluator:
PyPI `pycocotools 2.0.11` measured LM-O MUSE at **0.4713** and the
BOP-toolkit cocoapi fork at **0.4832**, vs public 0.4770 — a ±0.006
evaluator spread that affects LM-O but essentially not YCB-V
(0.6901 / 0.6902 / 0.6900). So "official LM-O MUSE = 0.471" is a *local
PyPI* figure, not the public row; label it accordingly whenever it is used
as a `muse-repro` baseline.
