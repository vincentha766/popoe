# Reproduction notes

This file is **not** a user guide and **not** a current performance claim.

There are **no frozen headline BOP numbers** in this repository yet. Do not cite internal or unpublished runs as popoe results. Numbers will be recorded here after a single, documented recipe is frozen and re-run end to end.

Until then:

* How the pipeline is factored: [ARCHITECTURE.md](ARCHITECTURE.md)
* How to install and run a clone: [README.md](README.md)
* Three identities (evaluated default vs paper-side flags vs parity oracle): [README.md](README.md#three-identities)
* BOP eval loop: `examples/bop_eval.py` — defaults are the **tuned Open3D** identity, not a paper-faithful freeze. Paper-side flags are on `--help`.
