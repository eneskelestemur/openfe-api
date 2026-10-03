# Plain MD

Plain MD runs a molecular dynamics simulation and hands back the trajectory. It computes no
free energy — OpenFE's `PlainMDProtocol` reports none by design — so it exists here for two
jobs:

- **Simulating a system on its own**: a predicted complex, the same protein without its
  ligand, a ligand in water.
- **Relaxing a structure before a free energy campaign**, which is the [`relax`](#relaxation)
  stage below.

## The input

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
    solvent: true          # the default

execution:
  repeats: 3               # independent runs, each from its own velocities
```

One system is one run, and `execution.repeats` independent simulations of it. A system needs
at least one molecule, and beyond that it is unconstrained:

| Shape | Allowed |
|---|---|
| protein + ligands + cofactors | yes |
| several ligands in one box | yes |
| protein alone | yes |
| ligand alone in water | yes |
| a charged molecule | yes |
| nothing at all | refused by name |

None of this is alchemical, so there is no single-ligand rule, no charge policy, no atom
mapping and no network. `cofactors` and `ligands` are treated identically; the two names only
keep a request readable.

!!! note "Vacuum needs a different nonbonded method"
    `solvent: false` runs in vacuum, which PME cannot do. Planning sets
    `forcefield_settings.nonbonded_method` to `nocutoff` and says so in the log. An explicit
    override of that setting wins over the automatic one.

## Cost

With OpenFE's defaults, one repeat is 0.1 ns NVT equilibration, 1 ns NPT equilibration and 5 ns
of production. The `screening` preset cuts the NPT equilibration to 0.2 ns and production to
0.5 ns, leaving NVT as it is. `plan` prints what the settings imply before anything is
submitted.

A plain MD run holds **one** GPU context, against 11 to 30 for the alchemical protocols, so it
packs onto a card far more readily and `submit` does not warn about `jobs_per_gpu`. Measured on
a 45,000 atom system, three simulations sharing one L40S peaked at 523 MiB between them,
against 6.9 GB for an RBFE complex phase and 13.6 GB for a SepTop solvent phase. Memory is
therefore not what limits packing here; compute is, and three per card is a reasonable starting
point.

## Results

```bash
openfe-api gather ./my_campaign
```

Gathering reports whether each repeat finished and where its files are: the production
trajectory, the topology, the minimized structure and the structures after NVT and NPT
equilibration. They are written to `results/simulations.tsv`, and over HTTP the paths come
back ready to fetch from [the files endpoint](service.md#getting-files-out).

There are deliberately no metrics here. RMSD, RMSF and pose stability belong to the analysis
stage, so that plain MD stays as simple as the protocol it wraps.

!!! note "A finished MD run writes a null estimate"
    `get_estimate()` returns None by design, so an MD repeat is judged finished by the
    trajectory its simulation unit reported, not by an estimate. Without that rule every repeat
    would look unfinished forever and `submit` would rerun completed work.

## Relaxation

A structure model's output carries strain a force field would not produce: bond geometry and
side chains that were never relaxed. Every protocol starts from that structure, and for a
series the reference pose also defines the frame every other ligand is placed into. Relaxing
once up front fixes the frame for the whole campaign, which a protocol's own pre-alchemical
equilibration cannot do, because that runs per edge and after placement.

Turn it on in any campaign, whatever protocol it runs:

```yaml
relax:
  enabled: true
  length: 1 nanosecond     # the NPT equilibration
  seed: 61453
```

Then the stage runs between `create` and `prep`:

```bash
openfe-api create my_request.yaml ./campaign
openfe-api relax  ./campaign --profiles profiles.yaml   # a GPU job
openfe-api prep   ./campaign                            # reads the relaxed frame
openfe-api plan   ./campaign
```

What it relaxes is whichever system carries the frame: **each complex** for ABFE and plain MD,
and **the reference complex** for RBFE or SepTop, since every other ligand is placed into the
reference's frame.

The frame it hands on is the structure after NPT equilibration, which the MD protocol already
writes with the water stripped. Relaxation is equilibration, so that is the frame; production
belongs to the standalone protocol.

On a real 420 residue complex, 0.05 ns of NPT moved the protein backbone by 0.65 A and the
ligand by 1.40 A in the protein's own frame, with its centroid moving 0.58 A: the pocket
relaxed and the pose stayed in it. There are no restraints, so that is the thing to check
rather than assume.

### Why it re-enters through preparation

The relaxed frame is not spliced into the prepared files. It is written to
`relaxed/<name>/system.pdb` with a map naming the residue each molecule landed in, and `prep`
reads it as an ordinary input structure. So preparation rebuilds every molecule from its
declared SMILES and re-checks heavy atoms, connectivity and stereochemistry against the relaxed
coordinates: a residue mapped wrongly fails by name instead of quietly relaxing into the wrong
chemistry. For a series the same pass re-places every other ligand onto the relaxed reference
pose, because that is what preparation does.

The map is not a convenience. OpenMM renumbers chains when it writes the system: on the complex
above the ligand went in as chain `B` and came out as chain `X`, so the request's own selectors
would have matched nothing.

Two consequences worth knowing:

- Preparation effectively runs twice, once inside the relaxation on the structure as supplied
  and once on the relaxed frame. The second takes seconds and is the one that validates.
- The checkpoint interval is 1 ns and a relaxation is usually shorter, so an interrupted
  relaxation starts over rather than resuming.

The relaxation inherits the campaign's partial charge settings and writes into the same charge
cache the campaign will use, so a relaxed campaign pays for its charges once.
