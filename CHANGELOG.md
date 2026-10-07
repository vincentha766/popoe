# Changelog

Notable changes to popoe. This file starts at the point the repository became a public tree; earlier history exists in git but was lab-internal and is not reconstructed here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html), with the caveat that `0.x` makes no compatibility promise: a stage Protocol may change in a minor release.

## [Unreleased]

Nothing yet.

## [0.1.0] — 2026-10-07

First public release. It is tagged but not published to PyPI, so there is no wheel to download; install from a clone.

The `main` branch was already public and untagged before this point, so the Changed, Removed and Fixed entries below are relative to that tree rather than to an earlier release.

Contracts and the fusion layer are CPU-tested. The reference run requires CUDA, GeDi, nvdiffrast, a BOP split, and detection JSONs, none of which are included in a clone.
There are no frozen headline BOP numbers in this release; see [README.md](README.md#three-identities) for what each configuration is.

### Added

- `PoseMethod.run(scene, obj)` as the library entry point, with two graphs behind it: `Pipeline` for correspondence-based methods, built by `make_correspondence_pipeline`, and `DirectPoseMethod` for external full-pose producers.
- Stage Protocols in `src/popoe/interfaces.py`: `Segmentor`, `QueryEncoder`, `TargetEncoder`, `PointDescriptor`, `FeatureFusion`, `PoseSolver`, `CoarseEstimator`, `PoseRefiner`, `GeometricRefiner`, `PoseScorer`, `Selector`, and `PoseMethod` itself.
  Conformance is structural, with no base class and no registry.
- FreeZe-v2 reference implementation under `src/popoe/freeze/`: DINOv2 visual features, a GeDi geometric branch, and `DinoGeDiFusion` as a separate component so query and target share one fitted visual PCA.
- Three `PoseSolver` implementations behind the same contract: `Open3DFeatureRansacSolver` (default), `GPURansacSolver` with selectable geometric or Eq. 5 feature fitness, and `TeaserSolver`.
- `GeometricRefiner` so `DirectPoseMethod` can run ICP without query or target features, with default clouds from CAD and depth in metres.
- `correspond_pair`, shared between `Pipeline.run` and `examples/bop_eval.py`, so both compute solve, refine and score identically.
- Config-addressed stage caching in `popoe.cache`, keyed by stage configuration, input content, and the keys of upstream fits.
- File-backed detection sources through one loader: official CNOS / CNOS-FastSAM, SAM-6D ISM, NIDS-Net, and MUSE, selectable by name and composable into a named-source union.
- A MUSE reimplementation (`popoe.segmentor_muse`) that acts as both a live segmentor and its own producer, since no official producer code is published. It writes `muse-repro`, never `muse`.
- `SAM6DPemResultsCoarseEstimator`, adapting external PEM pose output through the `CoarseEstimator` contract rather than as a `Segmentor`.
- Per-dataset BOP layouts in `popoe.datasets.bop.BOP_LAYOUTS`, shared by `bop_eval.py` and the local `metrics.ar` / `metrics.vsd` / `metrics.grasp` scorers. T-LESS and HB use `test_primesense`; ITODD uses gray `.tif`.
- `scripts/freeze_detections.py` as a detection-file identity gate, with `PROVENANCE.md` and `MANIFEST.sha256` per source directory.
- GitHub Actions CPU workflow: `pytest` with GPU extras skipped via `importorskip`.
- `CONTRIBUTING.md`, stating the constraints a patch has to respect alongside the failure each one prevents, and `CHANGELOG.md`.

### Changed

- `best_encoders()` now matches `bop_eval.py` in requiring nvdiffrast, so a missing GPU rasteriser is an error rather than a silent swap to trimesh with different CAD views.
- A stage whose backend is missing raises `BackendUnavailable` instead of substituting a weaker implementation under the same name. Substitution is the caller's policy, recorded in `Detection.source`.
- Documentation reorganised so each file has one subject: `README.md` for installing and running a clone, `ARCHITECTURE.md` for the seams, `docs/invariants.md` for the properties a running pipeline must satisfy, and `docs/sources/` for the four external producers. `docs/README.md` indexes them.
- The `Documentation` project URL resolves to `docs/` rather than `ARCHITECTURE.md`, and a `Source` URL is added alongside it.
- Prose rewritten for plain technical diction and sentence-length lines. No technical claim changed.

### Removed

- Experimental switches from the `bop_eval.py` CLI: `--probe-corr`, `--size-select`, `--size-select-no-fallback`, `--dual-assign`, `--corr-topk`, `--n-restarts`, `--max-targets`, `--score-coarse`, `--score-feat-w`.
  The underlying code remains in the library and in `scripts/`; it is no longer on the public runner.
- Lab-internal campaign notes, sibling-repository paths, and the campaign ledger from the public tree.
- `CNOS.md`, `MUSE.md`, `NIDS_NET.md` and `SAM6D.md` from the repository root, replaced by `docs/sources/` under one shared template.
- `REPRODUCTION.md`, which contained only links; its reproduction-status statement moved to `docs/README.md` and `README.md`.

### Fixed

- The detections loader was documented as raising "loudly" on a type error, which inverts the point. The failure it prevents is silent: `"1" in [1]` is false, so the image yields no candidates and appears to contain no instance of the object.
- `--topk` was documented as floored per class by `inst_count`, which reads as a cap. `bop_muse.py` computes `max(topk, inst_count)`, so a target with more instances than `--topk` is not truncated.
- Four rows of the stage table in `ARCHITECTURE.md` listed "class" or "function" where a Protocol exists: `FeatureFusion`, `CoarseEstimator`, `PoseScorer` and `Selector`.

[Unreleased]: https://github.com/vincentha766/popoe/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/vincentha766/popoe/releases/tag/v0.1.0
