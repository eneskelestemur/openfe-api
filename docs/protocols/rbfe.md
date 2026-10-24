# Relative binding free energy

A run here is an *edge*, and the per-ligand numbers are fitted over the whole network. Two
constraints follow, and both are enforced during preparation and planning: every ligand has
to sit in one protein frame, and a pair has to share enough core to map.

## The request

```yaml
protocol: rbfe
name: my_rbfe_campaign

reference: ligand_1             # defaults to the first entry in 'ligands'

protein:
  chains: [A]
  ph: 7.4

ligands:
  - name: ligand_1              # the reference: carries the protein and the trusted pose
    structure: ./lead_complex.cif
    selector: {chain: B, resname: LIG}
    smiles: "Cc1ccc(cc1)C(=O)Nc1ccccc1"
  - name: ligand_2              # SMILES only is enough
    smiles: "Clc1ccc(cc1)C(=O)Nc1ccccc1"
  - name: ligand_3
    path: ./docked_3.sdf        # a dedicated coordinate file, rather than a combined one
    smiles: "Fc1ccc(cc1)C(=O)Nc1ccccc1"
    pose: keep                  # overrides alignment.pose for this ligand

cofactors:
  - name: NAD
    selector: {chain: C}
    smiles: "NC(=O)c1ccc[n+](...)c1"

alignment:
  pose: mcs                     # mcs | keep
  n_conformers: 50
  max_core_rmsd: 0.5            # angstrom; a conformer worse than this is rejected
  rank_by: clash                # clash | energy

similarity:
  min_core_fraction: 0.6        # of the smaller molecule's heavy atoms
  allow_ring_break: false
  max_perturbed_atoms: 12       # optional
  core_smarts:                  # optional; skips the core search

network:
  method: minimal_redundant     # minimal_redundant | minimal_spanning | radial | explicit
  redundancy: 2
  mapper: lomap                 # lomap | kartograf | both

charges:
  correct_single: true
  allow_multi: false

settings:
  preset: default
execution:
  profile: cluster_gpu
  repeats: 3
```

The rules every protocol shares — paths, SMILES, stereochemistry, selectors — are in the
[input contract](../running/inputs.md).

## The reference frame

Structure models give each prediction its own protein coordinates, so one entry is the
**reference**: it supplies the protein, and its ligand pose is used exactly as given. Every
other prediction's protein is discarded.

The reference must carry the protein, in its own `structure` file or through `protein.path`.

## Placing each ligand

`alignment.pose: mcs`, the default, **builds each ligand on the common core**: conformers are
generated, superposed on the core, their core atoms moved onto the reference's, and the rest
relaxed around them. The best fit to the pocket is kept, landing within a few hundredths of
an angstrom of the reference's core. A ligand with no coordinates is fine.

!!! note "SMILES-only ligands need defined stereochemistry"
    Elsewhere an undefined stereocenter is taken from the pose. A SMILES-only ligand has no
    pose, so every center must be defined.

`pose: keep` uses the coordinates as supplied, which is what you want for a series docked
into one fixed receptor. Either value can be set per ligand, as `ligand_3` does above. A kept
pose whose core sits more than 2 Å from the reference's is warned about: mapped atoms
starting far apart converge poorly.

A ligand bringing its own predicted complex is still placed on the core, but its protein is
superposed onto the reference and the transplanted pose's core distance reported as
`predicted_core_rmsd`. A large value means the prediction disagrees with the series' binding
mode, or the core matched the wrong atoms.

## The similarity gate

Ligands are gated on how much core they share with the reference, not on scaffold identity.
A ligand that fails is excluded with a named reason in the preparation report, and the rest
of the series still prepares.

Naming `core_smarts` skips the core search, which can match the wrong rings on a symmetric
molecule. Murcko scaffolds are reported either way, and a series spanning more than one is
warned about: such a set is often better run as several campaigns, or with a radial network
per group.

Two ligands differing **only in stereochemistry** are dropped however good the mapping looks:
such a pair maps atom for atom, so its perfect score would make the planner prefer it over
every real edge. The message names [SepTop](septop.md), which can run the pair.

## The network

| method | what it does | when |
|---|---|---|
| `minimal_redundant` | cheapest connected network, then again, so every ligand has `redundancy` paths | production |
| `minimal_spanning` | cheapest connected network, N−1 edges | triage |
| `radial` | every ligand to one hub, set with `central_ligand` | a quick series around a lead |
| `explicit` | exactly the pairs in `edges` | reproducing a previous network |

Redundancy costs roughly twice the edges, and buys a network that survives a failed run plus
the cycles [cycle closure](#cycle-closure) needs.

`lomap` maps on 2D topology with a 3D distance cutoff. `kartograf` maps on geometry. `both`
keeps the better scoring mapping.

To rerun the same edges later, read the pairs out of `plans/network.json` and give them back
as `method: explicit` with an `edges` list.

## Charge changes

Only the *difference* across an edge matters. OpenFE reports it as state A minus state B, so
a ligand losing one unit of charge reads as `+1`.

| difference | default | cost |
|---|---|---|
| 0 | runs normally | one unit |
| ±1 | explicit charge correction, transforming a water into a counterion | ~4x: 22 lambda windows at 20 ns each |
| more | excluded from the network | n/a |

A forbidden pair is scored last so the planner routes around it, and any that survives is
removed afterwards. A ligand left with no path to the network is named. `allow_multi: true`
runs those edges and warns that the result is uncorrected.

## Cost

Per edge, per repeat, with OpenFE's defaults:

| Phase | Windows | Simulation |
|---|---|---|
| solvent | 11 | 66 ns |
| complex | 11 | 66 ns |
| total | | **132 ns** |

A charge-corrected edge runs about four times that, so it is submitted as its own Slurm array
with its own wall-time limit; see [execution](../running/execution.md#slurm). Two or three
edges share a 46 GB GPU comfortably.

## Results

`openfe-api gather` writes two tables:

- `results/edges.tsv` — each edge's complex phase minus its solvent phase. Both phases must
  have finished; an edge missing one is listed with the reason.
- `results/ligands.tsv` — per-ligand values fitted over the whole network.

!!! warning "Per-ligand values are relative"
    The fit is centered on the network's mean, so **differences between ligands are meaningful
    while a single value is not**. These are not absolute binding free energies.

`prep` prints how each ligand was placed and which were excluded; `plan` prints the edges with
their charge decisions and mapping scores. `--only` is rejected: a series is prepared as a
whole, and its runs are the edges the planner chose.

### Cycle closure

It also reports what each cycle in the network closes to:

| Cycle | closes to (kcal/mol) | per edge |
|---|---:|---:|
| `ligand_1` to `ligand_2` to `ligand_3` | +0.40 | +0.23 |

More than 1 kcal/mol is a failure. A network with no cycles reports `unknown` rather than
passing, since there was nothing to test.

### Connectivity

A failed edge can disconnect part of the network. Gathering names the ligands that lost
their path to the rest and leaves them out of the fit.
