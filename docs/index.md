# openfe-api

Run [OpenFE](https://docs.openfree.energy/) free energy experiments from protein-ligand
structures, on a Slurm cluster or a server VM. Usable as a Python library, a command line
tool, and a containerized HTTP service.

!!! warning "Early development"
    Every protocol is implemented and has run end to end on GPUs against a real series, but
    only at smoke lengths, so no production-length numbers have been produced yet.

## Why this exists

**An API, so jobs run without anyone driving them.** A long-running server accepts a
campaign, queues it on the cluster, and reports progress. The campaign directory is the
shared state, so work started through the service continues from the CLI, and the reverse.

**Agents can drive it.** A campaign is one validated document and each stage is one call.
The service publishes an OpenAPI schema generated from the same models the CLI uses, so a
tool discovers the contract rather than being taught it.

**Inputs are declared, not guessed.** Structure-model ligands carry no hydrogens and no bond
orders, and the SMILES written inside the file can be wrong. You supply the chemistry; the
structure supplies the pose. A run either starts from what you declared, or fails naming the
file, the residue and the rule that was broken.

## The pipeline

| Stage | Command | What it does |
|---|---|---|
| Prepare | `prep` | Rebuilds molecules from your SMILES, protonates the protein, checks it parameterizes |
| Plan | `plan` | Charges, chemical systems, transformations, and the cost before you commit |
| Execute | `submit` | One process per repeat, on Slurm or locally, resumable |
| Gather | `gather` | Repeats into free energies, checked against OpenFE's criteria |
| Analyze | `analyze` | Figures, correlation against measurement, trajectory metrics |

## Start here

- [Install](start/install.md), then [your first campaign](start/quickstart.md).
- [Choosing a protocol](protocols/index.md) compares ABFE, RBFE, SepTop and plain MD.
- [Service and container](running/service.md) covers the HTTP API and the images.
