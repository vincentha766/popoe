# Official NIDS-Net_WA_Sappe detections (source tag: `nids`)

BOP `method_info/601`, task "Model-based 2D segmentation of unseen objects".
Single batch 2024-05-27 (8980–8986) — all seven sets are segmentation; no
mixed-task trap here. LM-O/YCB-V were already in-tree and are byte-identical
to the official downloads (cross-checked 2026-07-26 and again 2026-08-06);
the other five downloaded 2026-08-06.

| Dataset | BOP submission | Records | SHA256 |
|---|---|---|---|
| LM-O | [8980](https://bop.felk.cvut.cz/sub_info/8980/) | 7179 | `8cf9c392a82153b3bbf1c6baa5a7a4fac056e6fc4f35ec645a1f3f76d6f75aea` |
| T-LESS | [8981](https://bop.felk.cvut.cz/sub_info/8981/) | 19714 | `16da4f7965e3adcaaa432163ba9f2953d42a3987aca1f38e7dcc42295901b11b` |
| TUD-L | [8982](https://bop.felk.cvut.cz/sub_info/8982/) | 12172 | `90137dcec2f140d2b8130e72524d751d1b94fd751efd90a05c8a089861357c4e` |
| IC-BIN | [8983](https://bop.felk.cvut.cz/sub_info/8983/) | 3873 | `2a39dad6d5273c45ef6c88415a78f30e7e6819bb654210a0917b8dcc1ca580cd` |
| ITODD | [8984](https://bop.felk.cvut.cz/sub_info/8984/) | 5410 | `cd3300ce053ee425be4b8bd9c003bfd4d08f2b6dc2153496d1ce74d5c57900dd` |
| HB | [8985](https://bop.felk.cvut.cz/sub_info/8985/) | 5589 | `1bac5e38fc97a6810c43adb6b733daa7ba533358a7e1c49773d543aff7f7a0d9` |
| YCB-V | [8986](https://bop.felk.cvut.cz/sub_info/8986/) | 12019 | `6eb751b20898e5cc8f499922590e9a07c2a645cfb7d5d14f7c59cb0d51c8544a` |

Files live in `outputs/seg_ap_20260725T223014Z/official_submissions/`;
symlinked here as `nids_wa_sappe_<ds>.json` (same naming as the two the
frozen Phase D recipes already reference).
