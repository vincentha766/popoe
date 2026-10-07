# popoe documentation

Four documents, each with one subject.

| Document | Subject |
|---|---|
| [../README.md](../README.md) | Installing a clone and running it: quickstart, external dependencies, the three identities, a BOP eval, consuming detections, writing a stage |
| [../ARCHITECTURE.md](../ARCHITECTURE.md) | The seams: the two method graphs, the stage protocols and their reference implementations, and why each boundary is where it is |
| [invariants.md](invariants.md) | Properties that must hold in a running pipeline, each with the failure that follows when it is removed |
| [sources/](sources/README.md) | The four external detection and pose producers: the producer boundary, source tags, environments, and per-producer setup |

[../scripts/README.md](../scripts/README.md) indexes the diagnostics and ablation drivers. They are not entry points for evaluation.

Outside this directory: [../CONTRIBUTING.md](../CONTRIBUTING.md) for submitting a patch, and [../CHANGELOG.md](../CHANGELOG.md) for what changed between versions.

## Which one to read

- Running the evaluated BOP loop, or installing the dependencies it needs: [../README.md](../README.md).
- Adding a solver, scorer, segmentor, or encoder: [../ARCHITECTURE.md](../ARCHITECTURE.md) for the contract, then [invariants.md](invariants.md) for the two rules every stage inherits.
- Obtaining or regenerating a detections file, or deciding which source tag an artefact belongs under: [sources/](sources/README.md).
- Investigating a result that looks wrong: [invariants.md](invariants.md) lists the silent failures the pipeline is built to prevent, which is where a wrong number usually originates.

Reproduction status: there are no frozen headline BOP numbers in this repository. Do not cite internal or unpublished runs as popoe results. See [../README.md](../README.md#three-identities) for what each configuration actually is.
