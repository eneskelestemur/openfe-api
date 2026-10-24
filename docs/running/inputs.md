# Input contract

A request names input files explicitly. Whole prediction folders are never accepted: you say
which file, which chain and which molecule, so nothing is resolved by guessing. Unknown
fields are rejected, so a typo fails loudly.

`protocol` selects the shape of the rest of the request — see
[ABFE](../protocols/abfe.md), [RBFE](../protocols/rbfe.md),
[SepTop](../protocols/septop.md) and [MD](../protocols/md.md) for each. This page covers the
rules all four share.

## Experiment, not hardware

The request describes the experiment: which molecules, which settings, how many repeats.
Where it runs and how repeats share a GPU belong to the
[execution profile](execution.md#execution-profiles) and nowhere else. `backend`, `mps` or
`jobs_per_gpu` under `execution` is an error naming the profile, because a setting given in
two places can be silently ignored in one of them.

Repeats come only from `execution.repeats`. OpenFE's own `protocol_repeats` is fixed at 1 and
cannot be overridden: each repeat is already its own process.

## Where the paths point

A path is a file on the machine running the stage: absolute, or relative to where you ran the
command. The CLI and `POST /campaigns` both read them that way.

Uploading is the exception. `POST /campaigns/upload` sends the files with the request, so
every path there is a **bare filename** matching an uploaded part, and the campaign keeps its
own copy under `inputs/`. See [uploading inputs](service.md#uploading-inputs).

## SMILES is the reference

Every ligand and cofactor needs a SMILES. It defines bond orders, formal charges and
protonation; the structure file supplies only coordinates.

!!! danger "A SMILES inside a structure file is not trusted"
    AlphaFold 3 writes a ligand SMILES into its mmCIF, and it can be wrong. One real example
    writes a nitro group as `[N+](=O)O`, which parses as a **+1 cation** — and ABFE rejects
    charged ligands, so a neutral molecule would lose its whole complex.

    Boltz-2 writes no SMILES at all, and inferring one from coordinates produces chemical
    nonsense: `[S+2]#C`, cumulated allenes, a carbene.

## Stereochemistry

Handled per center, not per molecule:

| Your SMILES | The pose | Result |
|---|---|---|
| undefined | either isomer | **Accepted.** Configuration taken from the pose and reported |
| `[C@H]` | `[C@H]` | Accepted |
| `[C@H]` | `[C@@H]` | **Rejected** — the model predicted the wrong isomer |

A center you defined must be reproduced; one you left undefined is taken from the pose and
reported. Chirality at phosphorus or sulfur bearing two or more terminal oxygens is skipped
in this comparison; sulfoxides are still compared.

## Selectors and duplicate copies

A ligand read from a combined structure file needs a `selector` resolving to exactly one
molecule. If several match, the error lists them.

When a combined file holds more than one copy of the ligand, as a homodimer does,
`extra_ligand_copies` says what happens to the extras:

- `drop` removes them, and excludes them from the protein too.
- `keep` retains them as non-alchemical components, present in both end states.

There is **no default**: the two choices model different things. The field is required
whenever the ligand comes from a combined file, and rejected when it has its own file.

## Charges

Each protocol has its own rule: [choosing a protocol](../protocols/index.md) compares them,
and each protocol page says where its rule is enforced. Cofactors may carry any charge under
all four.

`neutralize_ligands: true` lets preparation neutralize a charged ligand. It is off by
default, warns prominently, and refuses charges protonation cannot remove, such as a
quaternary nitrogen. The computed affinity then refers to the neutral form.

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

Clashes between ligand and protein are a warning, not an error: a slightly distorted pose is
often still usable.
