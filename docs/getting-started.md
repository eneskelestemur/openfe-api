# Getting started

## Install

openfe-api needs OpenFE 1.12 and its simulation stack, which come from conda-forge:

```bash
git clone https://github.com/eneskelestemur/openfe-api.git
cd openfe-api
conda env create -n openfe-api -f container/environment.yml
conda activate openfe-api
pip install -e '.[service]'          # add ,dev,docs to work on the project
```

Check the install:

```bash
openfe-api version
```

A [container image](service.md#container-images) carries the same environment, which is the
easier route on a cluster where you would rather not solve conda yourself.

## A first campaign

The repository ships a tiny campaign built from the test data. It is not a meaningful
calculation — the "protein" is a twelve residue fragment and the ligand is 1-phenylethanol —
but it exercises every stage and runs without a GPU.

```bash
openfe-api create examples/abfe_smoke_test.yaml ./smoke
openfe-api prep   ./smoke
openfe-api plan   ./smoke
openfe-api status ./smoke
```

## The stages

| Command | What it does | Cost |
|---|---|---|
| `create` | Validates the request and makes a campaign directory | instant |
| `relax` | Optional: queues a short MD relaxation of the inputs | minutes, on a GPU |
| `prep` | Builds protein, ligand and cofactors; checks them | seconds |
| `plan` | Assigns charges, writes transformations, estimates cost | seconds to minutes |
| `submit` | Runs or queues every repeat | hours to days |
| `gather` | Collects results and checks convergence | seconds |

`relax` only applies to a request that sets `relax.enabled`; see
[plain MD and relaxation](md.md). Every stage records what it did in the campaign directory,
so stages can run on different machines: plan on a login node, submit to a cluster, gather
wherever you like.

## The campaign directory

```text
my_campaign/
├── manifest.json        state of every run, plus provenance
├── inputs/              the campaign's own copy of uploaded input files
├── prepared/            protein.pdb, the molecules, prep_report.json, charges/
├── plans/               transformation JSON, network.json, network.graphml for RBFE
├── runs/<run>/repeatN/  working directory, results.json, quickrun.log
├── relaxed/<name>/      the optional relaxation: its input, frame and report
├── results/             results.tsv for ABFE, edges.tsv and ligands.tsv for a network
├── logs/                scheduler logs
└── submit_<class>.sh    the generated batch script, one per cost class
```

A **run** is a complex for ABFE, a system for plain MD, and an edge for RBFE or SepTop. ABFE
and MD prepare each run into `prepared/<name>/`; a series shares one frame, so it prepares into
`prepared/` directly with its ligands under `prepared/ligands/`.

A **task** is one repeat of one run. An RBFE edge is two tasks, one per phase, under
`runs/<edge>/solvent/` and `runs/<edge>/complex/`; everything else is one task per repeat,
because the protocol builds both phases itself.

The manifest is the source of truth. The CLI and the service both read and write it, so work
started through one is visible to the other.

`inputs/` only appears when a campaign was created through the service's upload endpoint;
everywhere else the request points at files that already live on disk. Either way the service
can list and serve anything in here, so a client that cannot see the filesystem still gets its
results. See [service and container](service.md#getting-files-out).

## Next steps

Write your own request by copying a template from `examples/`, then read the
[input contract](input-contract.md) for what each field means.
