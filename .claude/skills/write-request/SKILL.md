---
name: write-request
description: Write or fix an openfe-api request document (the YAML or JSON that defines a free energy campaign) — choosing the protocol, declaring ligand SMILES and selectors, handling duplicate copies, stereochemistry, charges and cofactors — and validate it before anything is run. Use this whenever the task involves creating, editing, reviewing or debugging an openfe-api request, converting an AlphaFold 3 or Boltz-2 prediction into a campaign input, or when a prep or validate step rejects a structure, ligand, selector or SMILES.
---

# Writing a request

A request is one document describing one experiment. It is validated as a whole before any
file is touched, so the fastest loop is: write it, run `openfe-api validate request.yaml`,
read the error, fix, repeat. Every rejection names the file, the residue and the rule.

## Start from a working example

Copy the closest file in `examples/` rather than writing from scratch — they are all valid
and are checked by the test suite:

| File | What it shows |
|---|---|
| `abfe_from_structure_model.yaml` | ABFE from one combined mmCIF |
| `abfe_split_files.yaml` | protein, ligand and cofactors as separate files |
| `rbfe_from_structure_models.yaml` | a series where each ligand has its own prediction |
| `rbfe_docked_series.yaml` | a series already docked into one receptor |
| `septop_scaffold_hop.yaml` | ligands too different to map |
| `md_complex.yaml`, `md_relax_then_abfe.yaml` | plain MD, and relaxation before a campaign |

Each protocol's page in `docs/protocols/` opens with a complete annotated request; read the
one you need and treat it as the field reference.

## The three rules that reject most first drafts

**Every ligand and cofactor needs a `smiles`.** It defines bond orders, formal charges and
protonation; the structure file supplies only coordinates. A SMILES written inside an mmCIF
is not trusted and is never read — AlphaFold 3 writes ones that parse as the wrong charge,
and Boltz-2 writes none. This is the single most common thing to get wrong when converting a
prediction into a request.

**A ligand in a combined file needs a `selector` that resolves to exactly one molecule.**
`{chain: E, resname: LIG_E}`. If it matches several, the error lists what matched.

**A combined file needs `extra_ligand_copies`.** Structure models frequently produce
homodimers. `drop` removes the extra copies and excludes them from the protein; `keep`
retains them as non-alchemical components present in both end states. There is no default,
because the two model different things.

## Choosing the protocol

`protocol` selects the shape of the rest of the document, and a field belonging to another
protocol is rejected rather than ignored. `docs/protocols/index.md` compares the four. In
short: ABFE for a number per ligand, RBFE for a congeneric series, SepTop when the
perturbation cannot be mapped, MD to simulate or to relax.

ABFE and MD list independent runs (`complexes:`, `systems:`). RBFE and SepTop describe one
series in one receptor (`ligands:` plus a `reference:`), and their runs are network edges
chosen during planning.

## Things that are decided per protocol

| | ABFE | RBFE | SepTop | MD |
|---|---|---|---|---|
| Ligand charge | must be neutral | any; a difference of one per edge is corrected | any, but all must match the reference | any |
| Fused aromatic rings | 3+ refused | fine | 3+ dropped | fine |
| Needs a reference pose | no | yes | yes | no |

`neutralize_ligands: true` lets preparation strip a charge where protonation can, warns, and
makes the affinity refer to the neutral form. It is off by default and should stay off
unless the user asks for it — it changes what was asked for.

## Stereochemistry

A center you define in the SMILES must match the pose, or the complex is refused. A center
you leave undefined is taken from the pose and reported. The exception: a SMILES-only ligand
in an RBFE series has no pose, so every center must be defined there.

## Where paths point

Paths are files on the machine running the stage, absolute or relative to the working
directory. The exception is the service's upload endpoint, where every path is a bare
filename matching an uploaded part — see the `drive-service` skill.

`structure:` takes a combined PDB or mmCIF. A ligand's own coordinate file (`.sdf`, `.mol`,
`.mol2`) goes in `path:`, not `structure:`.

## Hardware never goes in the request

`backend`, `mps` and `jobs_per_gpu` belong to the execution profile. The request carries
only `execution.profile` (which profile to use) and `execution.repeats`. A request that
sets hardware fields is rejected with a message naming the profile.

## Validate before handing it back

```bash
openfe-api validate request.yaml
```

This parses the document, resolves every path and checks the protocol's own rules without
reading a structure. A request that has not been validated is not finished. For the rules
each field is checked against, read `docs/running/inputs.md`.
