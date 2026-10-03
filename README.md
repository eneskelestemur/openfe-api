# openfe-api

Run [OpenFE](https://docs.openfree.energy/) free energy experiments from protein-ligand
structures, on a Slurm cluster or a server VM. Usable as a Python library, a CLI, and a
containerized FastAPI service.

**[Documentation](https://eneskelestemur.github.io/openfe-api/)**

Status: early development, version 0.1.0. Every protocol is implemented and has run end to end
on GPUs, but only at smoke lengths, so no production numbers have been produced yet. Structured
analysis of trajectories is the next milestone.

## Why

**An API, so jobs run without anyone driving them.** The library sits behind an HTTP service,
so a long-running server can accept a campaign, queue it on the cluster, and report its
progress. The campaign directory is the shared state, so work started through the service can
be inspected or continued from the CLI, and vice versa.

**Agents can drive it.** A campaign is one validated document and each stage is one call, which
is what an agent needs to run a free energy calculation without a person translating between
steps. The service publishes an OpenAPI schema generated from the same models the CLI uses, so
a tool discovers the contract rather than being taught it.

**Inputs are declared, not guessed.** Structure-model ligands carry no hydrogens and no bond
orders, and the SMILES written inside the file can be wrong. You supply the chemistry; the
structure supplies only the pose. A run either starts from what you declared, or it fails with
a message naming the file, the residue and the rule that was broken.

## The protocols

`protocol:` in the request selects one. The same commands run all four.

| Protocol | Answers | Reach for it when |
|---|---|---|
| `abfe` | absolute ΔG per ligand | you need a number per ligand, not a comparison |
| `rbfe` | ΔΔG per edge | the ligands are a congeneric series |
| `septop` | ΔΔG per edge | the perturbation is a scaffold hop RBFE cannot map |
| `md` | a trajectory | you want to simulate a system, or relax one first |

An `abfe` or `md` request lists independent runs; `rbfe` and `septop` describe one series in
one receptor, and their runs are the network edges chosen during planning. Any request may
also set `relax` to run a short MD of its input structure before preparation, which is worth
doing when the input came from a structure model.

## Install

The simulation stack is conda-only, so the environment comes first:

```bash
git clone https://github.com/eneskelestemur/openfe-api.git
cd openfe-api
conda env create -n openfe-api -f container/environment.yml
conda activate openfe-api
pip install -e '.[service]'
```

Requires Python 3.12+. A container image avoids solving the environment on every machine;
[container/](container/) has a Dockerfile and an Apptainer recipe that builds without Docker.
`openfe-api version` confirms the install.

To use it as a library from another project, install it into that project's environment —
which still needs the conda stack for anything that prepares or simulates:

```bash
pip install 'openfe-api @ git+https://github.com/eneskelestemur/openfe-api.git@v0.1.0'
```

The package ships type information, so your type checker sees the request models.

## Use

```bash
openfe-api validate examples/abfe_smoke_test.yaml     # check the request alone
openfe-api create   examples/abfe_smoke_test.yaml ./my_campaign
openfe-api prep     ./my_campaign                     # build protein, ligand, cofactors
openfe-api plan     ./my_campaign -p 8                # charges + transformation JSON
openfe-api submit   ./my_campaign --profiles profiles.yaml
openfe-api gather   ./my_campaign                     # results + quality checks
openfe-api status   ./my_campaign
```

Planning prints the simulation time the settings imply, so the cost is visible before anything
is submitted. Partial charges are cached per campaign, since AM1BCC takes minutes per
drug-sized molecule.

[examples/](examples/) holds a template per shape: a combined AF3 model file, a split-file
layout, RBFE from structure models or from a docked series, SepTop over scaffold hops, plain MD
of a complex and its apo protein, and a relaxed ABFE campaign.

## Execution

Each repeat runs as its own `openfe quickrun` process, so repeats spread across GPUs, pack onto
one GPU through CUDA MPS, and resume individually. `submit` writes a Slurm job array, or runs
locally across the detected GPUs; `--dry-run` writes the script without submitting it.
Rerunning `submit` after an interruption skips what finished and continues the rest from its
cache.

Where the work runs lives in an [execution profile](examples/profiles.yaml), kept separate from
the request so the same campaign is portable.

## Service and container

The same library is an HTTP service, reading and writing the same campaign directory as the
CLI, so the two are interchangeable.

```bash
uvicorn openfe_api.service:app --host 0.0.0.0 --port 8000
```

Interactive documentation is served at `/docs`; see [docs/service.md](docs/service.md) for the
endpoints, the environment variables, and the Docker and Apptainer images in
[container/](container/).

A client with no access to the service's filesystem can work entirely over HTTP:
`POST /campaigns/upload` sends the input files with the request, and `/campaigns/{name}/files`
lists and serves everything the campaign produced.

The service has no authentication yet, so reaching its port is as good as a shell as the user
running it. Keep it on localhost or behind something that authenticates; auth is next.

## Documentation

Published at <https://eneskelestemur.github.io/openfe-api/>, from the sources in
[docs/](docs/). To read it locally:

```bash
pip install -e '.[docs]'
mkdocs serve
```

## Development

```bash
pip install -e '.[service,dev,docs]'
pytest
```

[docs/contributing.md](docs/contributing.md) has the checks a change has to pass and the house
rules that shape most of them.

## Citation

If openfe-api is part of how you produced a result, please cite it. GitHub reads
[CITATION.cff](CITATION.cff), so the "Cite this repository" button gives you the current
version in BibTeX or APA. In BibTeX:

```bibtex
@software{openfe_api,
  author  = {Kelestemur, Enes},
  title   = {{openfe-api}: an {API} for running {OpenFE} free energy experiments on {HPC} and server {VMs}},
  version = {0.1.0},
  year    = {2026},
  license = {MIT},
  url     = {https://github.com/eneskelestemur/openfe-api}
}
```

Cite OpenFE and the protocols you ran alongside it; see below.

## Acknowledgements

openfe-api is a thin layer over other people's work. The science belongs to
**[OpenFE](https://docs.openfree.energy/)**
([source](https://github.com/OpenFreeEnergy/openfe)), which provides every protocol run here --
ABFE, RBFE, SepTop and plain MD -- together with their settings, atom mappings and results.
OpenFE is MIT licensed and so free to build on, but it is a scientific package first: if you
publish numbers produced through openfe-api, cite OpenFE and the protocols you ran as well.

The rest of the stack this depends on, directly or through OpenFE:

| Package | What it does here |
|---|---|
| [gufe](https://github.com/OpenFreeEnergy/gufe) | chemical systems, transformations, and the token registry the plans round-trip through |
| [OpenMM](https://openmm.org/) | the simulation engine every run ends up in |
| [openmmtools](https://github.com/choderalab/openmmtools) | alchemical factories and multistate samplers |
| [openmmforcefields](https://github.com/openmm/openmmforcefields) | small molecule force field templates |
| [OpenFF](https://openforcefield.org/) toolkit, units and [NAGL](https://github.com/openforcefield/openff-nagl) | ligand parametrization, quantities, and fast graph-net charges |
| [AmberTools](https://ambermd.org/AmberTools.php) | AM1BCC partial charges |
| [RDKit](https://www.rdkit.org/) | molecule building, the MCS search, and Open3DAlign pose placement |
| [PDBFixer](https://github.com/openmm/pdbfixer) | completing residues and adding hydrogens |
| [MDAnalysis](https://www.mdanalysis.org/) | trajectory handling |
| [LOMAP](https://github.com/OpenFreeEnergy/Lomap) and [Kartograf](https://github.com/OpenFreeEnergy/kartograf) | atom mappers and edge scoring |
| [konnektor](https://github.com/OpenFreeEnergy/konnektor) | the network generators behind RBFE planning |
| [PyMBAR](https://github.com/choderalab/pymbar) | the MBAR estimates and their uncertainties |
| [cinnabar](https://github.com/OpenFreeEnergy/cinnabar) | the maximum likelihood fit over a network |
| [pydantic](https://docs.pydantic.dev/) | the request schema and its error messages |
| [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/) | the CLI and its console output |
| [FastAPI](https://fastapi.tiangolo.com/) and [uvicorn](https://github.com/encode/uvicorn) | the HTTP service |

## License

MIT; see [LICENSE](LICENSE).
