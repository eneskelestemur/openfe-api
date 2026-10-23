# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-10-23

### Added

- `--trajectories` on `openfe-api analyze`, and the matching field on the endpoint, measure
  protein RMSD, ligand RMSD and ligand drift per repeat and aggregate them per run.
- Interaction fingerprints over the pocket, as occupancy per residue and interaction type,
  with every run in one heatmap.
- A representative pose per run in `analysis/scenes/`: the frame nearest the ligand's median
  position, as a PDB, a PyMOL script and a standalone py3Dmol page.
- `--workers` measures several trajectories at once, and `--submit` runs the analysis as a
  Slurm job, under the profile `--profile` names or the campaign's own.
- The overlap, exchange and convergence plots OpenFE writes are linked into `analysis/plots/`
  under names saying which run and repeat they came from.

### Changed

- An edge whose uncertainty comes out as zero, as one between two identical ligands does, no
  longer costs the network its per-ligand fit: the solver reads it as 0.001 kcal/mol and says
  so.
- A measured column that is all positive, or predictions that run against the measurements,
  are both called out: an affinity such as a pIC50 rises where a free energy falls.
- A trajectory that fails for any reason is reported and the rest are measured, where only
  our own errors were caught before.
- `metrics.json` is written on every analysis, so the versions behind the figures and tables
  are recorded even with nothing to correlate against.
- A symlink inside a campaign is served by the files endpoint; one pointing outside is not.
- `mdanalysis`, `netCDF4` and `openfe-analysis` are declared rather than relied on arriving
  with openfe, and `prolif` and `py3Dmol` are added for the interactions and the scenes.

## [0.2.0] - 2026-10-22

### Added

- `openfe-api analyze` and `POST /campaigns/{name}/analysis` write a campaign's figures and
  tables into `analysis/`. Read-only and repeatable: it never changes run states.
- For a relative protocol the predictions are shifted onto the measured mean, so the measured
  axis keeps real free energies. The shift is recorded.
- A correlation figure against measured free energies, with Pearson, Spearman, Kendall,
  MUE and RMSE, and a ranking figure for a campaign with nothing to compare against.
  Ligands with a prediction but no measurement are shown beside the correlation rather
  than dropped.
- Measured values are read from a CSV or TSV in kcal/mol. No affinity conversion is done:
  a ΔG table is required.
- `--dark` draws the figures for a dark background.

### Changed

- Campaigns gain an `analysis/` directory, served as a file group like the others.
- matplotlib is a new runtime dependency.

## [0.1.1] - 2026-10-16

### Fixed

- Placing a ligand on its common core failed for a series built on a bridged aromatic bicyclic,
  such as a pyrazolo[1,5-a]pyridine. MMFF typing kekulizes the molecule it is given and the
  aromatic flags do not come back, so the core query, which is written in aromatic bonds,
  stopped matching and every analogue was dropped with "the common core no longer matches after
  building the molecule". A core now matches whatever kekulé state the molecule is in.
- Shape alignment typed the reference ligand it was handed. A series reuses one reference for
  every ligand, so that left it kekulized for everything placed afterwards. It is typed on a
  copy, which O3A scores identically.
- A series with no edge to run named `prep_report.json` as the place to find the per-ligand
  reasons without having written it. The report is written before the refusal.

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

[Unreleased]: https://github.com/eneskelestemur/openfe-api/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/eneskelestemur/openfe-api/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/eneskelestemur/openfe-api/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/eneskelestemur/openfe-api/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/eneskelestemur/openfe-api/releases/tag/v0.1.0
