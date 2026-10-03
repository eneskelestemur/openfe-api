# Preparation

```bash
openfe-api prep ./my_campaign [--only NAME] [--skip-parameter-check]
```

Preparation turns your inputs into simulation-ready files and writes a report of everything it
did. A run that fails is recorded and the rest continue, so one bad input does not hide
problems in the others.

Each ABFE complex and each MD system is prepared independently, into `prepared/<name>/`. An
RBFE or SepTop series is instead prepared as a whole into one shared frame; see
[how each ligand is placed](rbfe.md#how-each-ligand-is-placed).

## Ligands and cofactors

1. Select the residue named by the selector and take its heavy atoms.
2. Apply your SMILES as the reference for bond orders, formal charges and protonation.
3. Read stereochemistry back from the 3D coordinates and compare it with the SMILES.
4. Add hydrogens with coordinates.
5. Check elements, radicals, net charge, and clashes against the protein.

The pose is never re-embedded or minimized: the coordinates you predicted are the
coordinates that get simulated.

## Proteins

PDBFixer completes the structure and adds hydrogens at the requested pH. Changes that
complete a structure are applied; changes that would alter the biology are refused:

| Situation | What happens |
|---|---|
| Missing side chain atoms | added, and reported |
| Missing hydrogens | added at the requested pH |
| Missing residues (chain gaps) | **not built** — warned about, gaps remain |
| Non-standard residues | **preparation stops**, they are listed |
| Terminal caps (ACE, NME) | kept as part of the chain |
| Waters | kept only if `keep_waters` is set |
| Recognized ions | kept |
| Anything else unrecognized | dropped, with a warning naming the residue |

Loops are not modeled because modeled loops are unreliable, and silently inventing
structure is worse than leaving a gap you know about. Non-standard residues stop
preparation rather than being substituted, because swapping a residue changes the protein.

Hydrogens are placed with OpenMM's CPU platform, so preparation behaves the same on a
login node, a CPU node or a GPU node, regardless of which GPU or OpenCL drivers are present.

## The force field check

Each prepared protein is parameterized with the Amber force field on the CPU. It costs about
a second and catches missing templates, bad protonation and broken residues **before** any
GPU time is spent. `--skip-parameter-check` turns it off.

## Outputs

```text
prepared/<run>/
├── protein.pdb
├── ligand.sdf              # ABFE; a plain MD system writes one SDF per molecule
├── cofactor_<name>.sdf
└── prep_report.json
```

The report records the prepared SMILES, net charges, atom counts, dropped residues, every
warning, and the versions of the tools used.

A net charge of zero is required only by ABFE, whose protocol refuses a charged alchemical
ligand. The other three accept any charge; see the [input contract](input-contract.md#charges).