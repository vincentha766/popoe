# Reproduction notes

This file is **not** a user guide and **not** a current performance claim.

popoe is still being cleaned up as a public library. Headline BOP numbers will be recorded here after a single, documented recipe is frozen and re-run end to end — not from the lab campaign history that used to live in this path.

Until then:

* How the pipeline is factored: [ARCHITECTURE.md](ARCHITECTURE.md)
* How to install and run a clone: [README.md](README.md)
* BOP eval loop: `examples/bop_eval.py` — defaults are the **tuned Open3D** identity, not a paper-faithful freeze. Paper-side flags are on `--help`.

Git history of this file still holds the older campaign tables if they are needed internally. Do not cite those figures as the project's public result.
