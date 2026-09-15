# Official SAM-6D detections (source tag: `sam6d`)

BOP method 441 segmentation files. The method page also lists 2D detection and 6D localization submissions — always check the Task field. Download each segmentation submission from the `sub_info` page ("Download submission") and save as `sam6d_official_<ds>.json` in this directory.

Two other filenames live here and must not be mixed with the official set:

- `sam6d_official_<ds>.json` — method 441 segmentation files.
- `sam6d_ism_{lmo,ycbv}.json` — local ISM runs (FastSAM proposals). Not byte-identical to any official file.

## Method 441 (`sam6d_official_<ds>.json`)

| Dataset | BOP submission | Records | SHA256 |
|---|---|---|---|
| LM-O | [6967](https://bop.felk.cvut.cz/sub_info/6967/) | 21264 | `638a933c0f3f404086f975050524ead00b23f6c081d77a1dce99443fab781108` |
| TUD-L | [6965](https://bop.felk.cvut.cz/sub_info/6965/) | 29482 | `267784437d15d97061dc30248bacdb08631780385fc7b86507818fd7ef63a6ab` |
| T-LESS | [6966](https://bop.felk.cvut.cz/sub_info/6966/) | 61082 | `e63e91376d3c116ea39aec2b5c173b0358f099fbb260275ab69fe48200e6fdf6` |
| IC-BIN | [6968](https://bop.felk.cvut.cz/sub_info/6968/) | 7778 | `3e4797bfda1dc2ca7514018ed082b6c8678351bf6905f2803725825bc167cef8` |
| ITODD | [6969](https://bop.felk.cvut.cz/sub_info/6969/) | 18915 | `d0511f138d0e509ee3fb028e5d3c438fa1f2cb6ae27e1ef4fc6d22b3968595e2` |
| HB | [6971](https://bop.felk.cvut.cz/sub_info/6971/) | 20142 | `f22f496109341f8bb0f03c0d33476bb0af69f468f604afbd7fc03c898dc2d39a` |
| YCB-V | [6970](https://bop.felk.cvut.cz/sub_info/6970/) | 46382 | `2288e24bfcbed29aedb719b53bff40f1a558a47f02392d5da0b6dabb2539abf8` |

## Local ISM runs (`sam6d_ism_{lmo,ycbv}.json`)

| File | SHA256 |
|---|---|
| `sam6d_ism_lmo.json` | `19f44ba740e422d3b7ad09d08656bcca03092a4dc1e21707d5f35243e49f1107` |
| `sam6d_ism_ycbv.json` | `dcadea8f62d37779747c52e153180c36d72b37fde3a3cd08ba36d02d75ef081c` |

## Test fixture (not a detection input)

| File | SHA256 |
|---|---|
| `union_cnos_sam6d_lmo.reference.json` (`tests/test_union_reference_xval.py`) | `5c11cf2d5d98db241798f55976c7cdcff9561350cd87cc46040747e599a7c40b` |

Schema per record: `scene_id, image_id, category_id, bbox, score, time, segmentation`. Verify with `python scripts/freeze_detections.py --check`.
