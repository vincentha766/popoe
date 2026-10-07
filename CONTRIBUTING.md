# Contributing

popoe is research code at `v0.1`, maintained by one person. Issues and pull requests are welcome; please read this first, because a few of the constraints here are unusual and most review comments trace back to them.

## Before writing code

Open an issue first for anything beyond a localised fix. A patch that changes a stage contract, adds a dependency, or alters what the evaluated default computes needs agreement on the approach before the implementation, since reverting one of those costs more than discussing it.

Read [docs/invariants.md](docs/invariants.md). It lists the properties a running pipeline must satisfy and the failure that follows when one is removed.
Several of them are easy to break with a change that appears correct when read in isolation, so a review comment citing one of those entries is not a style objection.

## Pull requests

**Scope.** One concern per pull request. A bug fix that also reformats the file, or a feature that also renames adjacent symbols, is harder to review and harder to revert than the two changes separately.

**Tests.** The suite must pass on CPU:

```bash
pip install -e ".[dev,reference]"
pytest tests/
```

A behaviour change needs a test that fails before the change and passes after.
GPU, nvdiffrast, GeDi, OpenCV, Open3D and TEASER++ are absent in CI, so a test that requires one calls `pytest.importorskip("...")` instead of asserting the package is present.
If a test fails, check whether it also fails on `main` before treating it as yours.

**Commit messages.** An imperative subject under 72 characters, stating what the commit does rather than what the problem was: `Share correspond_pair between Pipeline and bop_eval`, not `fix duplication`.
Add a body wherever the motivation is not evident from the diff, covering the previous behaviour and what fails without the change.

**Documentation.** Each document has one subject, listed in [docs/README.md](docs/README.md). A change that belongs in two of them usually belongs in one with a cross-reference. Keep lines under roughly 300 characters and break paragraphs at sentence boundaries, so a diff isolates one claim.

## Constraints specific to this project

**No silent backend substitution.** A stage whose backend is missing raises `BackendUnavailable`. It must not fall back to a weaker implementation under the same name, however reasonable the substitute.
Two methods behind one name make results unattributable and corrupt the config-addressed cache. Substitution is the caller's policy, composed explicitly and recorded in `Detection.source`.

**Every feature-changing parameter belongs in the cache key.** `popoe.cache` is config-addressed. A new encoder parameter that is not part of the key turns a sweep into a replay over stale features, which produces plausible numbers rather than an error. Adding a parameter means adding it to the key.

**Source tags are reserved.** `cnos`, `sam6d`, `nids` and `muse` name artefacts published by those authors. A reimplementation writes its own tag, such as `cnos-lab` or `muse-repro`. Do not widen a tag to cover locally produced output.

**No bare `except` in the eval loop.** A swallowed exception becomes zero rows, indistinguishable from "object not found".

**Units.** Mesh vertices in millimetres; unprojected depth and pose `t` in metres. BOP CSVs convert back to millimetres at the edge. A conversion added anywhere else is almost certainly wrong.

**Performance claims.** Do not add BOP numbers to the repository.
There is no frozen headline result yet, and the three configurations documented under [README.md](README.md#three-identities) are easily confused: the evaluated default is a tuned Open3D recipe rather than a paper-faithful reproduction.
A benchmark result in a pull request description is useful as evidence. A number committed to a document is a claim the project is not yet ready to make.

**External producers stay external.** CNOS, NIDS-Net, SAM-6D and MUSE run in their own environments and write files that popoe reads. Do not add their dependencies to `pyproject.toml` or import them from `popoe`. See [docs/sources/](docs/sources/README.md).

## Adding a stage implementation

Conformance is structural: match the Protocol's method signature in [src/popoe/interfaces.py](src/popoe/interfaces.py). There is no base class and no registry.
[ARCHITECTURE.md](ARCHITECTURE.md) documents each stage and its reference implementation, and [README.md](README.md#writing-a-stage) has a minimal solver.

A new implementation should reuse the existing scoring and refinement stages rather than reimplementing them. If the pipeline cannot express what the implementation needs, that is worth an issue — the seam may be in the wrong place.

## Reporting a bug

Include the command, the dataset and split, the detections file with its SHA256, and the relevant `POPOE_*` variables. A cache directory reused across a parameter change is a common cause of a result that cannot be reproduced, so state whether `--cache` was fresh.
