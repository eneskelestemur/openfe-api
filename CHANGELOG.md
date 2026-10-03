# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-10-03

First release. This entry describes what the project does.

### Protocols

- **ABFE** (`protocol: abfe`): absolute binding free energy per ligand, with a Boresch
  restraint search validated before any GPU time is spent.
- **RBFE** (`protocol: rbfe`): relative binding free energies over a congeneric series, with
  network planning (star, spanning tree, cyclic and redundant variants), per-edge atom
  mappings, and cycle-closure checks on the gathered result.
- **SepTop** (`protocol: septop`): relative binding free energies across scaffold hops that
  RBFE cannot map, with hierarchical pose placement (`auto`, `keep`, `mcs`, `shape`) falling
  back to Open3DAlign, and a screening gate with configurable thresholds and a drop-or-warn
  policy.
- **Plain MD** (`protocol: md`): a trajectory for a protein, a complex or ligands in solvent,
  reporting whether each repeat finished and where its artifacts are.
- **Relaxation** (`relax:` on any request): an optional short MD of the input structure before
  preparation, which takes the NPT frame forward. Worth doing when the input came from a
  structure model.

### Input and preparation

- A strict, explicit input contract: ligand chemistry is declared, never guessed from a
  structure file's own SMILES. Structure-model outputs (AlphaFold 3, Boltz-2) are supported as
  first-class inputs.
- Protein cleanup, ligand building, cofactor handling and a force field parametrization check
  that runs before submission.
- Partial charges cached per campaign, keyed by SMILES and method.

### Execution

- Each repeat runs as its own `openfe quickrun` process, so repeats spread across GPUs, pack
  onto one GPU through CUDA MPS, and resume individually.
- Slurm job arrays or local execution across detected GPUs, selected by an execution profile
  kept separate from the request. `--dry-run` writes the script without submitting it.
- Rerunning `submit` after an interruption skips what finished and continues the rest.

### Interfaces

- A Python library, a Typer CLI (`create`, `relax`, `prep`, `plan`, `submit`, `gather`,
  `status`, `validate`, `version`), and a FastAPI service that reads and writes the same
  campaign directory, so the two front ends are interchangeable.
- `POST /campaigns/upload` creates a campaign from uploaded input files, for a client with no
  access to the service's filesystem. The campaign keeps its own copy of each file under
  `inputs/`.
- `GET /campaigns/{name}/files`, `/files/{path}` and `/archive` read a campaign back out over
  HTTP, so the whole cycle works against a container on a machine you cannot log into. Single
  files support ranged requests; the archive is an uncompressed tar that reports its length up
  front and leaves simulation output out unless asked for.
- Docker and Apptainer images carrying both entry points.

[Unreleased]: https://github.com/eneskelestemur/openfe-api/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/eneskelestemur/openfe-api/releases/tag/v0.1.0
