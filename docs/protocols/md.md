# Plain MD

`PlainMDProtocol` reports no free energy, so this protocol hands back trajectories and files
rather than numbers. It covers two jobs: simulating a system on its own, and
[relaxing a structure](#relaxation) before a free energy campaign.

## The request

```yaml
protocol: md
name: my_md_campaign

systems:
  - name: holo
    structure: ./af3_model.cif
    protein: {chains: [A], ph: 7.4}
    ligands:
      - name: lig
        selector: {chain: E, resname: LIG_E}
        smiles: "Cc1ccc(cc1)C(=O)Nc1ccccc1"
    cofactors:
      - name: NAD
        selector: {chain: C}
        smiles: "NC(=O)c1ccc[n+](...)c1"
    solvent: true               # false runs in vacuum

settings:
  preset: default
execution:
  repeats: 3                    # independent runs, each from its own velocities
```

The rules every protocol shares are in the [input contract](../running/inputs.md).

## What a system may hold

One system is one run. It needs at least one molecule and is otherwise unconstrained:
protein with ligands and cofactors, several ligands in one box, a protein alone, a ligand
alone in water, a charged molecule. An empty system is refused by name.

Nothing is alchemical, so there is no single-ligand rule, no charge policy, no atom mapping
and no network. `cofactors` and `ligands` are treated identically; the two names only keep a
request readable.

!!! note "Vacuum needs a different nonbonded method"
    `solvent: false` runs in vacuum, which PME cannot do. Planning sets
    `forcefield_settings.nonbonded_method` to `nocutoff` and says so. An explicit override of
    that setting wins.

## Cost

One repeat is 0.1 ns NVT equilibration, 1 ns NPT equilibration and 5 ns production. The
`screening` preset cuts NPT to 0.2 ns and production to 0.5 ns.

A plain MD run holds **one** GPU context, against 11 to 30 for the alchemical protocols, so
`submit` does not warn about `jobs_per_gpu` and three per card is a reasonable start.

## Results

`openfe-api gather` reports whether each repeat finished and where its files are: the
production trajectory, the topology, the minimized structure, and the structures after NVT
and NPT equilibration. They go to `results/simulations.tsv`, and over HTTP the paths come back ready
to fetch from [the files endpoint](../running/service.md#getting-files-out).

Metrics belong to [analysis](../results/analysis.md), not here.

!!! note "A finished MD run writes a null estimate"
    `get_estimate()` returns None by design, so an MD repeat is judged finished by the
    trajectory its simulation unit reported. Without that rule every repeat would look
    unfinished and `submit` would rerun completed work.

## Relaxation

`relax` runs a short MD of the input structure before preparation, so the whole campaign is
placed into one relaxed frame rather than into a structure model's raw output. Turn it on in
any campaign, whatever protocol it runs:

```yaml
relax:
  enabled: true
  length: 1 nanosecond          # the NPT equilibration
  seed: 61453
```

The stage then runs between `create` and `prep`:

```bash
openfe-api relax ./campaign --profiles profiles.yaml   # a GPU job
openfe-api prep  ./campaign                            # reads the relaxed frame
```

It relaxes whichever system carries the frame: **each complex** for ABFE and plain MD, **the
reference complex** for RBFE or SepTop. The frame handed on is the structure after NPT
equilibration, water stripped.

On a real 420 residue complex, 0.05 ns of NPT moved the protein backbone by 0.65 Å and the
ligand by 1.40 Å in the protein's own frame, with its centroid moving 0.58 Å. There are no
restraints, so that is the thing to check rather than assume.

### What to know about it

The relaxed frame is written to `relaxed/<name>/system.pdb` with a map naming the residue
each molecule landed in, and `prep` reads it as an ordinary input structure. So every
molecule is rebuilt from its declared SMILES and re-checked against the relaxed coordinates,
and for a series every other ligand is re-placed onto the relaxed reference pose.

The map matters: OpenMM renumbers chains when it writes the system, so a ligand that went in
as chain `B` can come out as chain `X`, and the request's own selectors would match nothing.

- Preparation effectively runs twice, once inside the relaxation and once on the relaxed
  frame. The second takes seconds and is the one that validates.
- The checkpoint interval is 1 ns and a relaxation is usually shorter, so an interrupted
  relaxation starts over rather than resuming.
- The relaxation writes into the campaign's own charge cache, so charges are paid for once.
