# Official MUSE detections (source tag: `muse`)

BOP `method_info/873`, task "Model-based 2D segmentation of unseen objects". **Nothing in popoe writes this name** — the reimplementation writes `source="muse-repro"`.

All seven BOP-Classic-Core mask files are publicly downloadable from the `sub_info` page ("Download submission"). Save as `muse-full_<ds>-test.json` in this directory.

| Dataset | BOP submission | Records | SHA256 |
|---|---|---|---|
| LM-O | [29108](https://bop.felk.cvut.cz/sub_info/29108/) | 7146 | `55061983089d6236c19cb9b6a8a6c754388d146287be45ec40ceb9c32dbe3003` |
| IC-BIN | [29109](https://bop.felk.cvut.cz/sub_info/29109/) | 4731 | `34a2a40b3c716bb3c36b0739d49ebc885019cb6d97079d4e5e1ba9c743ed1427` |
| TUD-L | [29110](https://bop.felk.cvut.cz/sub_info/29110/) | 15736 | `38dc40cfa75f22a74f1f85cb10fb2283adb99db65ea496dff68bf216beeccb8b` |
| T-LESS | [29111](https://bop.felk.cvut.cz/sub_info/29111/) | 26511 | `78bcdab72d0eac44ab5b8477eec9e229fdaa2e61fdc69bcec48be46a3f230482` |
| ITODD | [29112](https://bop.felk.cvut.cz/sub_info/29112/) | 6320 | `2d34ebce3a464f129f6cdc8770686df56869eafa8bd8fff135fbfecb3c65813a` |
| HB | [29063](https://bop.felk.cvut.cz/sub_info/29063/) | 6440 | `c0e0802a3db1e2394507099098ed5000208d93e1701f8b19850d6cd6d7d59d1d` |
| YCB-V | [29113](https://bop.felk.cvut.cz/sub_info/29113/) | 16902 | `b4703a218d13f707d47556b2733eeddc38fea7d89bf927d113da25349c74f497` |

Every record carries `scene_id, image_id, category_id, bbox, score, time, segmentation`. Follow the `sub_info` page's "Download submission" link, then `python scripts/freeze_detections.py --check`.

HB is `29063`, not in the contiguous 29108–29113 block — do not infer IDs by counting. The same method page also lists a **detection** (bbox-only) batch for the same seven sets; those files have no `segmentation` field and must not go in this directory.
