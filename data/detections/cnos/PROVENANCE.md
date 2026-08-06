# Official CNOS-FastSAM detections (source tag: `cnos`)

The seven BOP-Classic-Core "default detections" for CNOS-FastSAM — the
official artefacts every arm's `cnos=` source points at (byte-identical to
the published files; batch **4003-4009** on the CNOS method page, which also
carries a detection-only batch 4017-4023 — same dual-batch trap as SAM6D 546
and MUSE 873, always check the Task field).

| Dataset | Records | SHA256 |
|---|---|---|
| LM-O | 13269 | `1a03d3c7a1d57a9c7e6e1bc162f99281b5044ca50428c619477ec4ab11fa375a` |
| T-LESS | 58667 | `db010fbce92149a54ae7a252176d6dee80823353a7e5d704c0f33657c5b1ecec` |
| TUD-L | 16889 | `400978b21a94aaa109d6e5039df7aefa7cdbdc6af037cbd1b05ab586ae6d540d` |
| IC-BIN | 6361 | `922b9878b1e8e8cac7d9245daa672de7568408ca0d4a8f9a7884bb532f93bcc3` |
| ITODD | 10617 | `cce4bcc9d33618e215f1099f9ac7f04598c0f39188585e739dd992496c3bbbd6` |
| HB | 13254 | `7eb39ad0d82783dc59a49cd2f6654c99b63d3b3ef3f051f3368056755e94e6b0` |
| YCB-V | 30602 | `fdec15729676e15876302fc620f752cc5290ee28da5fc3c7e17da1072fd4f422` |

Verification and pod-side materialization: `scripts/freeze_detections.py`
(`--check` refuses symlinks and unregistered hashes). Always JSON-parse after
any download — size alone lies (see the HB lesson in sam6d/PROVENANCE.md).
