# openfe-api

Run [OpenFE](https://docs.openfree.energy/) free energy experiments from protein-ligand
structures, on a Slurm cluster or a server VM. Usable as a Python library, a command line
tool, and a containerized HTTP service.

!!! warning "Early development"
    Every protocol is implemented: absolute (ABFE), relative (RBFE) and separated topologies
    (SepTop) binding free energies, and plain MD. Each has run end to end on GPUs against a
    real series, but those runs were short plumbing checks, so no production-length numbers
    have been produced yet. Structured analysis is planned.

## Why this exists

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

## What it does

<div class="grid cards" markdown>

-   __Prepare__

    Recovers bond orders from your SMILES, protonates the protein, handles cofactors and
    duplicate chains, and checks the result parameterizes before any GPU time is spent.

-   __Plan__

    Builds the chemical systems, assigns partial charges once and caches them, validates
    against the protocol, and tells you the simulation cost before you commit.

-   __Execute__

    Runs each repeat as its own process on Slurm or locally, packs repeats onto GPUs with
    CUDA MPS, and resumes cleanly after a wall-time limit.

-   __Gather__

    Combines repeats into free energies and checks convergence against OpenFE's own
    criteria, or collects the trajectories a plain MD run produced.

</div>

## The four protocols

| Protocol | Answers | Reach for it when |
|---|---|---|
| [ABFE](preparation.md) | absolute ΔG per ligand | you need a number per ligand, not a comparison |
| [RBFE](rbfe.md) | ΔΔG per edge | the ligands are a congeneric series |
| [SepTop](septop.md) | ΔΔG per edge | the perturbation is a scaffold hop RBFE cannot map |
| [Plain MD](md.md) | a trajectory | you want to simulate a system, or relax one first |

## Where to go next

- [Getting started](getting-started.md) runs a whole campaign end to end.
- [Input contract](input-contract.md) is the reference for what a request must contain.
- [Service and container](service.md) covers the HTTP API and images.
