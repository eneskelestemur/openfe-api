# Relative binding free energy

RBFE compares ligands to each other rather than measuring each one on its own. Instead of one
calculation per ligand, you get one per *edge* between two similar ligands, and the per-ligand
numbers come from fitting those edges together.

That makes it cheaper per answer and usually more accurate, but it adds two constraints ABFE
does not have: the ligands must be similar enough to mutate between, and they must all occupy
one protein frame.

## One frame, from one reference

An edge mutates ligand A into ligand B *inside a single shared protein*. The protein is one
component across the whole campaign, so every ligand has to sit in one coordinate frame.

Structure models give you the opposite: each prediction has its own protein coordinates. So one
entry is the **reference**. It supplies the protein, and its ligand pose is the one pose used
exactly as given. Every other prediction's protein is discarded.

```yaml
reference: ligand_1     # defaults to the first entry in 'ligands'
```

The reference must carry the protein, either in its own `structure` file or through
`protein.path`.

## How each ligand is placed

Because the other proteins are discarded, moving a predicted pose into the reference frame
would put it in a receptor it was never predicted against — the side chains it was optimized
for are gone. RBFE convergence depends instead on mapped core atoms starting in consistent
relative geometry across the whole network, and pose-to-pose noise between close analogues is
exactly that inconsistency.

So by default each non-reference ligand is **built on the common core**: conformers are
generated, each is superposed on the core, its core atoms are moved onto the reference's, and
the rest of the molecule relaxes around them. The one fitting the pocket best is kept. In
practice the core lands within a few hundredths of an angstrom of the reference's.

```yaml
alignment:
  pose: mcs             # the default
  n_conformers: 50
  max_core_rmsd: 0.5    # angstrom; a conformer worse than this is rejected
  rank_by: clash        # clash | energy
```

A ligand with no coordinates at all is therefore fine — SMILES is enough.

!!! note "SMILES-only ligands need defined stereochemistry"
    Elsewhere, a stereocenter the SMILES leaves undefined is taken from the pose. A
    SMILES-only ligand has no pose to take it from, so every center must be defined.
    Enumerating them instead would quietly multiply the size of your campaign.

### Keeping a docked pose

If you docked the series into one fixed receptor, the poses are already in the right frame and
regenerating them would throw the docking away. Set `pose: keep`:

```yaml
alignment:
  pose: keep
```

Individual ligands can override the campaign default either way:

```yaml
ligands:
  - name: ligand_4
    smiles: "Clc1ccc(cc1)C(=O)Nc1ccccc1"
    pose: mcs           # no docked pose for this one
```

A kept pose whose core sits more than 2 Å from the reference's is warned about, because mapped
atoms starting far apart converge poorly.

### What happens to a prediction you supply

A ligand that brings its own predicted complex is still placed on the core by default, but the
prediction is not thrown away: its protein is superposed onto the reference and the transplanted
pose's core distance is reported as `predicted_core_rmsd`. A large value means either the
prediction disagrees with the series' binding mode, or the common core matched the wrong atoms.
It costs nothing and it is the cheapest sanity check available on a predicted series.

## The similarity gate

An edge needs a common core to mutate through, so ligands are gated on how much of that core
they share with the reference — not on scaffold identity. A ring-size change alters a scaffold
but is a perfectly routine edge, so scaffold-based filtering would throw away valid work.

```yaml
similarity:
  min_core_fraction: 0.6      # of the smaller molecule's heavy atoms
  allow_ring_break: false
  max_perturbed_atoms: 12     # optional
```

A ligand that fails is excluded with a named reason in the preparation report, and the rest of
the series still prepares.

If you already know the core, name it and skip the search. This is more reliable than a search,
which can match the wrong rings on a symmetric molecule:

```yaml
similarity:
  core_smarts: "O=C(Nc1ccccc1)c1ccccc1"
```

Murcko scaffolds are still computed and reported. A series spanning more than one scaffold is
warned about: relative results across chemotypes are less trustworthy, and such a set is often
better run as several campaigns, or with a radial network per group.

## Choosing the edges

```yaml
network:
  method: minimal_redundant   # the default
  redundancy: 2
  mapper: lomap               # lomap | kartograf | both
```

| method | what it does | when |
|---|---|---|
| `minimal_redundant` | cheapest connected network, then again, so every ligand has `redundancy` paths | production |
| `minimal_spanning` | cheapest connected network, N−1 edges | triage |
| `radial` | every ligand to one hub, set with `central_ligand` | a quick series around a lead |
| `explicit` | exactly the pairs in `edges` | reproducing a previous network |

Redundancy costs roughly twice the edges and buys two things: the network survives a failed run,
and its cycles give you **cycle closure**, the only quality check RBFE has that needs no
experimental data.

`mapper: lomap` maps on 2D topology with a 3D distance cutoff and is the default.
`kartograf` maps on geometry. `both` offers each and keeps the better scoring mapping.

### Reproducing a network exactly

Planning writes `plans/network.graphml` and `plans/network.json`. To rerun the same edges —
after changing settings, or to compare against an older campaign — read the edge list out of
`network.json` and give it back as `explicit`:

```yaml
network:
  method: explicit
  edges:
    - [ligand_1, ligand_2]
    - [ligand_2, ligand_3]
```

## Charge changes

RBFE keeps ligand charges as given; only the *difference* across an edge matters. OpenFE reports
that difference as state A minus state B, so a ligand losing one unit of charge reads as `+1`.

```yaml
charges:
  correct_single: true     # the default
  allow_multi: false       # the default
```

| difference | default | cost |
|---|---|---|
| 0 | runs normally | one unit |
| ±1 | explicit charge correction, transforming a water into a counterion | ~4x: 22 lambda windows at 20 ns each |
| more | excluded from the network | n/a |

A pair the policy forbids is scored last so the planner routes around it, and any that survives
is removed afterwards. If that leaves a ligand with no path to the rest of the network, it is
named rather than silently producing a number you cannot compare. `allow_multi: true` runs those
edges anyway and warns that the result is uncorrected.

Because a corrected edge costs around four times a neutral one, it is submitted as its own
Slurm array with its own wall-time limit. See [Execution](execution.md).

## Pairs RBFE will not run

Two ligands that differ **only in stereochemistry** are dropped, however good the mapping
looks. A hybrid topology maps such a pair atom for atom, so the edge returns a free energy near
zero with convincing statistics instead of failing — and a perfect mapping score makes the
planner prefer it over every real edge. The message names [SepTop](septop.md), which needs no
mapping and can run the pair.

## Results

Gathering an RBFE campaign gives two tables:

- `results/edges.tsv` — the relative binding free energy of each edge, which is its complex
  phase minus its solvent phase. Both phases must have finished; an edge missing one is listed
  with the reason rather than half an answer.
- `results/ligands.tsv` — per-ligand values fitted over the whole network.

!!! warning "Per-ligand values are relative"
    The fit is centered on the network's mean, so **differences between ligands are meaningful
    while a single value on its own is not**. These are not absolute binding free energies.

### Cycle closure

The free energies around a cycle must sum to zero. What they actually sum to is the error the
network carries, and it needs no experimental data to compute:

| Cycle | closes to (kcal/mol) | per edge |
|---|---:|---:|
| `ligand_1` to `ligand_2` to `ligand_3` | +0.40 | +0.23 |

A cycle closing to more than 1 kcal/mol is reported as a failure. A network with no cycles
reports the check as `unknown` rather than passing it, because there was nothing to test — one
more reason to prefer `minimal_redundant`.

### Connectivity

A failed edge is not an isolated loss the way a failed ABFE complex is: it can disconnect part
of the network, and anything downstream of it loses its path to the rest. Gathering names those
ligands explicitly and leaves them out of the fit.

## Worked commands

```bash
openfe-api validate examples/rbfe_smoke_test.yaml
openfe-api create   examples/rbfe_smoke_test.yaml ./my_series
openfe-api prep     ./my_series
openfe-api plan     ./my_series
openfe-api submit   ./my_series
openfe-api gather   ./my_series
```

`prep` prints how each ligand was placed and which were excluded; `plan` prints the edges with
their charge decisions and mapping scores; `gather` prints the edges, the fitted ligand values
and the cycle closures.

`--only` is rejected here: a series is prepared as a whole, and its runs are the edges the
planner chose.
