# Separated topologies

A run here is an edge, as in [RBFE](rbfe.md), but with no atom mapping to satisfy — so the
request has no similarity gate and the network has no mapping score to optimize. What it has
instead is placement: with nothing to map, a badly placed ligand gives a confident wrong
answer rather than a failure, so preparation reports how every ligand was placed and how
well.

## The request

```yaml
protocol: septop
name: my_septop_campaign

reference: lead                 # defaults to the first entry in 'ligands'

protein:
  chains: [A]
  ph: 7.4

ligands:
  - name: lead                  # the reference: carries the protein and the trusted pose
    structure: ./lead_complex.cif
    selector: {chain: B, resname: LIG}
    smiles: "Cc1ccc(cc1)C(=O)Nc1ccccc1"
  - name: saturated_hop         # no shared core; placed by shape
    smiles: "O=C(Nc1ccncc1)C1CCOCC1"

cofactors:
  - name: NAD
    selector: {chain: C}
    smiles: "NC(=O)c1ccc[n+](...)c1"

alignment:
  pose: auto                    # auto | keep | mcs | shape
  auto_core_fraction: 0.6
  n_conformers: 50
  rank_by: clash                # clash | energy

screening:
  policy: warn                  # warn | drop
  max_size_ratio: 1.5           # heavy-atom ratio to the reference, either direction
  max_clashes: 5                # placed heavy atoms within the clash cutoff of the protein
  min_core_fraction:            # off by default

network:
  method: radial                # radial | radial_redundant | explicit
  central_ligand:               # defaults to the reference

settings:
  preset: default
execution:
  profile: cluster_gpu
  repeats: 3
```

Cofactors appear in both end states of every edge, and are dropped for the solvent phase,
which holds only the two ligands. The rules every protocol shares are in the
[input contract](../running/inputs.md).

## Placing each ligand

`auto` resolves per ligand and reports what it chose:

| Situation | Path | What happens |
|---|---|---|
| The ligand supplies coordinates | `keep` | Used as given; a file carrying its own protein is superposed onto the reference first |
| It shares a core of at least `auto_core_fraction` | `mcs` | Conformers are built, placed on that core, and relaxed around it |
| Anything else | `shape` | Open3DAlign overlays it on the reference ligand's shape and chemical features |

Open3DAlign matches atoms by chemical features rather than a shared substructure, which is
what makes a hop placeable. Preparation reports its score and the matched-atom RMSD per
ligand, in `prep_report.json` and the `prep` table. The score grows with molecule size, so it
compares conformers of one ligand, not different ligands.

Forcing `pose: mcs` on a ligand sharing fewer than three atoms with the reference is refused:
two atoms fix a position but not an orientation.

!!! note "Shape overlap is not a docking result"
    An overlaid pose is a plausible starting point in the right pocket. If you have docked
    poses, supply them and keep them.

## Screening

Two exclusions have no knob, because the protocol itself refuses the work:

- **A net charge change.** `SepTopProtocol` rejects any difference between its end states and
  there is no correction scheme. In a star network every edge touches the reference, so a
  ligand whose charge differs from the reference's cannot appear at all.
- **A ligand OpenFE cannot restrain**, meaning three or more fused aromatic rings. Restraint
  atoms are picked for *both* ligands of an edge, so the
  [limitation](abfe.md#rules-of-its-own) applies here too, and such a ligand is dropped
  during planning.

The rest is thresholds plus `policy`, which decides whether a ligand failing one is warned
about or dropped. `min_core_fraction` is unset by default: refusing a ligand for sharing no
core would refuse exactly the hops this protocol exists for.

## The network

With no per-pair score to minimize, the network is a **star on the reference**: every ligand
is one edge from the ligand whose pose you trust.

- `radial` is N−1 edges, the cheapest connected network.
- `radial_redundant` adds edges between consecutive spokes, closing a cycle through the hub.
  Roughly double the cost, and it buys [cycle closure](rbfe.md#cycle-closure). The extra
  pairs follow the order the ligands were requested in.
- `explicit` runs exactly the pairs you list.

RBFE's `minimal_spanning` and `minimal_redundant` are not offered: with no score they would
build an arbitrary tree while looking optimized.

No `network.graphml` is written — a SepTop edge carries no atom mapping. `plans/network.json`
is the record, and `method: explicit` reproduces those edges.

## Cost

Per edge, per repeat, with OpenFE's defaults:

| Phase | Windows | Simulation |
|---|---|---|
| solvent | 27 | 299 ns |
| complex | 19 | 211 ns |
| total | | **510 ns** |

Both phases run inside one `openfe quickrun` process, so an edge is one task rather than two
as an RBFE edge is. SepTop is the heaviest protocol here for GPU memory, at roughly 13.5 GB
per job; two edges share a 46 GB card. See
[sharing a GPU](../running/execution.md#sharing-a-gpu-with-cuda-mps).

## Results

`openfe-api gather` reads each edge's single result file, whose estimate already carries
both ligands' standard state corrections, and fits the network exactly as RBFE's is. The
output is the same two tables, [`edges.tsv` and `ligands.tsv`](rbfe.md#results), with
per-ligand values relative to the network mean.

The uncertainty of a single repeat is the two phases' MBAR errors in quadrature, because the
file's own top-level uncertainty is the spread across OpenFE's internal repeats, of which
openfe-api always runs exactly one.
