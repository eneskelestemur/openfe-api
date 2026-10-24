# Preparation and planning

```bash
openfe-api prep ./my_campaign [--only NAME] [--skip-parameter-check]
openfe-api plan ./my_campaign [--only NAME] [-p 8]
```

`prep` turns the request's inputs into simulation-ready files and a report. `plan` assigns
partial charges, builds the chemical systems, validates them against the protocol, and writes
the transformation JSON that `submit` runs. A run that fails is recorded and the rest
continue.

Each ABFE complex and MD system is prepared independently into `prepared/<name>/`. An RBFE or
SepTop series is prepared as a whole into one shared frame; see
[how each ligand is placed](../protocols/rbfe.md#placing-each-ligand).

## Ligands and cofactors

1. Select the residue named by the selector and take its heavy atoms.
2. Apply your SMILES as the reference for bond orders, formal charges and protonation.
3. Read stereochemistry back from the 3D coordinates and compare it with the SMILES.
4. Add hydrogens with coordinates.
5. Check elements, radicals, net charge, and clashes against the protein.

The pose is never re-embedded or minimized: the coordinates you predicted are the coordinates
that get simulated.

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

Hydrogens are placed on OpenMM's CPU platform, so preparation behaves the same on a login
node as on a GPU node.

Each prepared protein is then parameterized with the Amber force field on the CPU. It costs
about a second and catches missing templates, bad protonation and broken residues **before**
any GPU time is spent. `--skip-parameter-check` turns it off.

## What preparation writes

```text
prepared/<run>/
├── protein.pdb
├── ligand.sdf              # ABFE; a plain MD system writes one SDF per molecule
├── cofactor_<name>.sdf
└── prep_report.json
```

The report records the prepared SMILES, net charges, atom counts, dropped residues, every
warning, and the versions of the tools used.

## The chemical systems

| Protocol | Transformations per run | End states |
|---|---|---|
| ABFE | 1 | protein + ligand + cofactors + solvent, and the same without the ligand |
| RBFE | 2, one per phase | one ligand each, with an atom mapping between them |
| SepTop | 1, covering both phases | one ligand each, both holding the protein |
| Plain MD | 1 | one system, used as both end states |

Cofactors and kept ligand copies appear unchanged in every state; only the alchemical ligand
differs.

## Partial charges

Charges are assigned once per molecule and cached in the campaign, keyed by SMILES and
method, so every repeat shares identical charges and re-planning is free.

!!! tip "AM1BCC is slow"
    The default took **9.5 minutes** for one drug-sized ligand plus two NAD cofactors, even
    across eight processes. NAGL takes seconds and is reasonable for screening:

    ```yaml
    settings:
      overrides:
        partial_charge_settings.partial_charge_method: nagl
    ```

## Settings presets

`default` keeps OpenFE's own defaults; `screening` shortens the simulations for triage.
Neither changes the lambda schedules. Anything else is overridden by dotted path:

```yaml
settings:
  preset: default
  overrides:
    thermo_settings.temperature: 310 kelvin
    complex_simulation_settings.production_length: 5 nanosecond
```

Overrides are checked against the settings model, so a misspelled path fails at planning time
with the available names listed. Values with units are written as strings.

## Knowing the cost first

`plan` prints the simulation time the settings imply, before anything is submitted:

```text
1,472 ns total (3 repeat(s) x 491 ns:
  complex 30 windows / 336 ns, solvent 14 windows / 155 ns)
```

Per protocol: [ABFE](../protocols/abfe.md#cost), [RBFE](../protocols/rbfe.md#cost),
[SepTop](../protocols/septop.md#cost), [plain MD](../protocols/md.md#cost).
