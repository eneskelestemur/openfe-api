# Input contract

A request names input files explicitly. Whole prediction folders are never accepted: you
say which file, which chain and which molecule, so nothing is resolved by guessing.

`protocol` selects the shape of the rest of the request, and a field belonging to another
protocol is rejected rather than ignored. This page covers `abfe` and the rules every protocol
shares. The others describe a series or a set of systems instead of independent complexes:
[rbfe](rbfe.md), [septop](septop.md), [md](md.md).

## A complete request

```yaml
protocol: abfe
name: my_abfe_campaign

complexes:
  - name: ligand_1
    structure: ./model.cif            # combined file, or per-component paths below
    protein:
      chains: [A, B]                  # omit to keep every chain
      ph: 7.4
      keep_waters: false
    ligand:
      selector: {chain: E, resname: LIG_E}
      smiles: "c1ccc(cc1)C(=O)NC2CCCCC2"
    extra_ligand_copies: drop         # drop | keep
    cofactors:
      - name: NAD_C
        selector: {chain: C}
        smiles: "NC(=O)c1ccc[n+](...)c1"

settings:
  preset: default                     # default | screening
  overrides:
    thermo_settings.temperature: 310 kelvin

execution:
  profile: cluster_gpu                # sets the backend and GPU packing
  repeats: 3                          # each repeat runs as its own process

neutralize_ligands: false

relax:                                # optional, any protocol
  enabled: false                      # a short MD of the input before preparation
```

Unknown fields are rejected, so a typo fails loudly rather than being ignored. `relax` is
available to every protocol; see [plain MD and relaxation](md.md#relaxation).

## What belongs in the request, and what does not

The request describes the **experiment**: which molecules, which protocol settings, how many
repeats. Where it runs and how repeats share a GPU depend on the **hardware**, so they are
set once, on the [execution profile](execution.md#execution-profiles), and nowhere else.

Putting `backend`, `mps` or `jobs_per_gpu` under `execution` is an error that names the
profile as the right place — a setting that could be given in two places could be silently
ignored in one of them.

Likewise the number of repeats comes only from `execution.repeats`. OpenFE's own
`protocol_repeats` setting is fixed at 1 and cannot be overridden, because each repeat is
already its own process; raising it would multiply the work of every one of them.

## Where the paths point

A path in a request is a file on the machine that runs the stage: absolute, or relative to
where you ran the command. The CLI and `POST /campaigns` both read them that way.

Uploading is the exception. `POST /campaigns/upload` sends the files with the request, so
there every path is a **bare filename** matching one of the uploaded parts, and the campaign
keeps its own copy under `inputs/`. A path with a directory in it, or one that does not match
an upload, is refused. See [uploading inputs](service.md#uploading-inputs).

## SMILES is the reference

Every ligand and cofactor needs a SMILES. It defines bond orders, formal charges and
protonation; the structure file supplies only coordinates.

!!! danger "A SMILES inside a structure file is not trusted"
    AlphaFold 3 writes a ligand SMILES into its mmCIF, but it can be wrong. One real
    example writes a nitro group as `[N+](=O)O`, which parses as a **+1 cation** — and the
    ABFE protocol rejects charged ligands, so the whole complex would be refused for a
    molecule that is really neutral.

    Boltz-2 writes no SMILES at all. Inferring one from the coordinates produces chemical
    nonsense: `[S+2]#C`, cumulated allenes, a carbene. There is no safe automatic path,
    which is why the SMILES is required from you.

## Stereochemistry

Handled per center, not per molecule:

| Your SMILES | The pose | Result |
|---|---|---|
| undefined | either isomer | **Accepted.** Configuration taken from the pose and reported |
| `[C@H]` | `[C@H]` | Accepted |
| `[C@H]` | `[C@@H]` | **Rejected** — the model predicted the wrong isomer |

An undefined center means the stereochemistry is unknown, as when a racemic mixture is given
to a structure model, so the prediction decides it. A center you did define must be
reproduced, or the pose is wrong and the complex is refused.

Chirality at phosphorus or sulfur bearing two or more terminal oxygens is ignored in this
comparison: the oxygens of a phosphate are equivalent by resonance, so a SMILES assigns
that center arbitrarily. Sulfoxides, which are genuinely stereogenic, are still compared.

## Selectors and duplicate copies

A ligand read from a combined structure file needs a `selector` that resolves to exactly
one molecule. If it matches several, the error lists them so you can narrow it.

Structure models frequently produce homodimers, giving two copies of the ligand. You must
say what happens to the extra copies:

- `drop` removes them, and they are excluded from the protein too.
- `keep` retains them as non-alchemical components, present in both end states.

There is **no default**. A homodimer is never resolved by guessing, because the two choices
model different things. The field is required whenever the ligand comes from a combined
file, and rejected when it has its own file, where extra copies cannot exist.

## Charges

Each protocol has its own rule, taken from what the protocol itself accepts:

| Protocol | Rule |
|---|---|
| ABFE | the alchemical ligand must be neutral |
| RBFE | any charge; a *difference* of one across an edge is corrected, more is refused |
| SepTop | any charge, but every ligand must match the reference's: no change is supported |
| Plain MD | any charge; nothing is alchemical |

Cofactors may carry any charge under all four.

The ABFE rule is checked by preparation and again by the protocol. The RBFE and SepTop rules
act where the pairs are known: RBFE drops the edge during planning, and SepTop drops the ligand
during preparation, because in a star network every edge touches the reference.

`neutralize_ligands: true` lets preparation neutralize a charged ligand. It is off by
default, warns prominently, and refuses charges that protonation cannot remove, such as a
quaternary nitrogen. Use it knowing the computed affinity then refers to the neutral form,
not the species you supplied.

## Everything that is checked

| Rule | Failure |
|---|---|
| SMILES parses | named, with the offending string |
| Heavy atom count matches the pose | both counts reported |
| Connectivity matches | says the counts agree but the bonding does not |
| Defined stereochemistry matches | prints both SMILES |
| Elements covered by the force field | lists the unsupported elements |
| No radical electrons | names the atoms |
| Ligand is neutral, for ABFE | gives the charge and the remedy |
| Selector resolves to one molecule | lists what matched |
| Referenced files exist | lists the missing paths |
| Protein parameterizes | reports the force field error |
| The restraint search can handle the ligand | names the OpenFE limitation (ABFE, SepTop) |
| A pair is not two stereoisomers | drops the edge and names SepTop (RBFE) |

Clashes between the ligand and the protein are reported as a warning rather than an error,
since a slightly distorted pose is often still usable.
