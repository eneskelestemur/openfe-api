# Separated topologies

SepTop answers the same question as [RBFE](rbfe.md) — how much better is this ligand than that
one — without needing an atom mapping between them. Both ligands sit in the pocket at the same
time, each restrained to the protein, and the calculation decouples one while coupling the
other.

That removes RBFE's hardest constraint. A hybrid topology can only mutate what it can map, so a
changed ring system, a different core or a ligand half the size of the reference is out of
reach. SepTop runs all of those.

Two things come with that freedom, and both shape how openfe-api uses the protocol:

- **Every edge costs the same**, about 510 ns per repeat, whether the two ligands differ by a
  methyl or share nothing but a binding site.
- **The poses carry the result.** With no mapping to fall back on, a ligand placed badly in the
  pocket gives a confident wrong answer rather than a failure.

## The input

A SepTop campaign looks like an RBFE one: one reference supplying the protein and the trusted
pose, then the ligands.

```yaml
protocol: septop
name: my_septop_campaign

reference: lead

protein:
  chains: [A]

ligands:
  - name: lead
    structure: ./lead_complex.cif
    selector: {chain: B, resname: LIG}
    smiles: "Cc1ccc(cc1)C(=O)Nc1ccccc1"
  - name: saturated_hop
    smiles: "O=C(Nc1ccncc1)C1CCOCC1"
```

Cofactors work as they do everywhere else: present in both end states of every edge, and
dropped automatically for the solvent phase, which holds only the two ligands.

## How each ligand is placed

`alignment.pose` defaults to `auto`, which resolves per ligand and reports what it chose:

| Situation | Path | What happens |
|---|---|---|
| The ligand supplies coordinates | `keep` | Used as given; a file carrying its own protein is superposed onto the reference first |
| It shares a common core of at least `auto_core_fraction` | `mcs` | Conformers are built, placed on that core, and relaxed around it |
| Anything else | `shape` | Open3DAlign overlays it on the reference ligand's shape and chemical features |

```yaml
alignment:
  pose: auto              # auto | keep | mcs | shape
  auto_core_fraction: 0.6
  n_conformers: 50
  rank_by: clash          # clash | energy
```

Open3DAlign matches atoms by their chemical features rather than by a shared substructure,
which is what makes a hop placeable at all. Preparation reports its score and the RMSD of the
atoms it matched, per ligand, in `prep_report.json` and in the `prep` table. The score grows
with molecule size, so it compares conformers of one ligand rather than different ligands.

Forcing `pose: mcs` on a ligand that shares fewer than three atoms with the reference is
refused by name: two atoms fix a position but not an orientation, so there is nothing to place
on.

!!! note "Shape overlap is not a docking result"
    An overlaid pose is a plausible starting point in the right pocket, not a docked pose. If
    you have docked poses, supply them and keep them.

## Screening

Two rules have no knob, because the protocol itself refuses the work:

- **A net charge change is excluded.** `SepTopProtocol` rejects any difference between its end
  states, and there is no correction scheme. In a star network every edge touches the
  reference, so a ligand whose charge differs from the reference's cannot appear at all.
- **A ligand OpenFE cannot restrain is excluded.** The complex phase picks Boresch restraint
  atoms automatically, for *both* ligands of an edge. In OpenFE 1.12 that search fails on
  ligands with three or more fused aromatic rings, so such a ligand is dropped during planning
  with the reason named.

The rest is about whether a pose is plausible, so it is thresholds plus a policy:

```yaml
screening:
  policy: warn            # warn | drop
  max_size_ratio: 1.5     # heavy-atom ratio to the reference, either direction
  max_clashes: 5          # placed heavy atoms within the clash cutoff of the protein
  min_core_fraction:      # off by default
```

`min_core_fraction` is deliberately unset: refusing a ligand for sharing no core would refuse
exactly the hops this protocol exists for. It is there for the case where you want SepTop's
accuracy on a congeneric series and want the gate anyway.

## The network

SepTop's cost does not depend on the pair, and it has no mapping score, so there is no per-pair
difficulty for a spanning tree to minimize. What is left is graph statistics, where a **star on
the reference** is the best tree: every ligand is one edge from the ligand whose pose and
affinity you know, and every edge contains that trusted pose.

```yaml
network:
  method: radial          # radial | radial_redundant | explicit
  central_ligand:         # defaults to the reference
```

- `radial` is N−1 edges, the cheapest connected network.
- `radial_redundant` adds edges between consecutive spokes, closing a cycle through the hub.
  That roughly doubles the cost and buys [cycle closure](rbfe.md#results), the only quality
  check needing no experimental data. The extra pairs follow the order the ligands were
  requested in, because no score here would justify choosing differently.
- `explicit` runs exactly the pairs you list.

RBFE's `minimal_spanning` and `minimal_redundant` are deliberately not offered. Without a
meaningful score they would build an arbitrary tree while looking like an optimized one, and an
arbitrary tree has long paths that accumulate error for the same number of edges.

No `network.graphml` is written, unlike RBFE: a SepTop edge carries no atom mapping, so the file
would hold empty ones. `plans/network.json` is the record, and `network.method: explicit`
reproduces exactly those edges.

## Cost and GPU packing

Per edge, per repeat, with OpenFE's defaults:

| Phase | Windows | Simulation |
|---|---|---|
| solvent | 27 | 299 ns |
| complex | 19 | 211 ns |
| total | | **510 ns** |

The `screening` preset cuts that to about 140 ns for triage.

Both phases run inside one `openfe quickrun` process, because the protocol builds them from a
single pair of end states. That makes an edge one task, as an ABFE complex is, rather than two
as an RBFE edge is.

It also makes SepTop the heaviest protocol here for GPU memory. Each multistate phase leaves
roughly one leaked context per window behind, so the solvent phase alone reaches 27 — more than
any other protocol's phase. On a real edge with a 420 residue protein that phase peaked at
13.6 GB, and the complex phase, with 19 windows, stayed below it.

The cache is released between the two phases, which was measured: GPU memory fell from 13.6 GB
to 435 MiB at the boundary, so a process holds the worse phase's contexts rather than both
phases' at once.

That still leaves 27 contexts per process, so two SepTop processes ask one device for 54, past
the 48 CUDA MPS is documented to serve, and `submit` warns about it. It was measured anyway:
**two edges sharing one 46 GB L40S completed with no CUDA error**, peaking at 26.8 GB, and took
26 minutes against 26 minutes for the same two edges on two GPUs. So packing doubles throughput
per card for a few percent of wall time, and what actually limits it is memory — about 13.5 GB
per job — rather than the context ceiling. **Two jobs per GPU needs roughly 28 GB; use one on a
16 GB card.**

## Results

`gather` reads each edge's single result file, whose top-level estimate is already the relative
binding free energy with both ligands' standard state corrections applied, and fits the network
exactly as RBFE does:

```bash
openfe-api gather ./my_campaign
```

The output is the same two tables — `results/edges.tsv` and `results/ligands.tsv` — with
per-ligand values relative to the network mean rather than absolute. The uncertainty of a single
repeat is the two phases' MBAR errors in quadrature, because the file's own top-level
uncertainty is the spread across OpenFE's internal repeats, of which openfe-api always runs
exactly one.

## When to use which protocol

| | ABFE | RBFE | SepTop | [MD](md.md) |
|---|---|---|---|---|
| Answers | absolute ΔG | ΔΔG per pair | ΔΔG per pair | a trajectory |
| Needs an atom mapping | no | yes | no | no |
| Handles a scaffold hop | yes | no | yes | n/a |
| Net charge change | must be neutral | one, corrected | not supported | any |
| Cost per answer | ~484 ns | ~130 ns | ~510 ns | ~6 ns |
| Measured peak VRAM | ~8.4 GB | ~6.9 GB | ~13.6 GB | ~0.2 GB |
| Jobs per GPU, measured | 1 | 2 or 3 on 46 GB | 2 on 46 GB | 3+ |

For a congeneric series, RBFE is four times cheaper and needs no shape heuristics. Reach for
SepTop when the mapping is the thing standing in your way.
