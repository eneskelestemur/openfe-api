# Planning

```bash
openfe-api plan ./my_campaign [--only NAME] [-p 8]
```

Planning assigns partial charges, builds the chemical systems, validates them against the
protocol, and writes the transformation JSON files.

For RBFE and SepTop it also chooses which ligand pairs to run; see
[relative binding free energy](rbfe.md) and [separated topologies](septop.md).

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
method, so every repeat shares identical charges and re-planning costs nothing.

!!! tip "AM1BCC is slow"
    The default, AM1BCC, took **9.5 minutes** for one drug-sized ligand plus two NAD
    cofactors, even across eight processes. NAGL takes seconds and is a reasonable choice
    for screening:

    ```yaml
    settings:
      overrides:
        partial_charge_settings.partial_charge_method: nagl
    ```

## Settings presets

`default` keeps OpenFE's own defaults; `screening` shortens the simulations for triage rather
than for numbers you would publish. Neither preset changes the lambda schedules, because the
window count must keep matching the schedule length. Anything else is overridden by dotted
path:

```yaml
settings:
  preset: default
  overrides:
    thermo_settings.temperature: 310 kelvin
    complex_simulation_settings.production_length: 5 nanosecond
```

The number of repeats is not a setting: it comes from `execution.repeats`, and
`protocol_repeats` is always 1. Overriding it is rejected.

Overrides are checked against the settings model, so a misspelled path fails at planning
time with the available names listed. Values with units are written as strings.

## Ligands the restraint search cannot handle

ABFE and SepTop pick Boresch restraint atoms automatically, and OpenFE 1.12's search fails on
any ligand with **three or more fused aromatic rings** — anthracene, phenanthrene, carbazoles,
acridines, pyrrolo-quinoxalines. Planning calls OpenFE's own function up front, so such a
ligand is refused with an explanation rather than failing after the complex phase has started
on a GPU, and the check passes again once OpenFE is fixed. ABFE refuses the complex; SepTop
drops the ligand from the network.

## Knowing the cost first

Planning prints the simulation time the settings imply, before anything is submitted:

```text
1,472 ns total (3 repeat(s) x 491 ns:
  complex 30 windows / 336 ns, solvent 14 windows / 155 ns)
```

The default ABFE protocol is expensive: 30 lambda windows in the complex leg and 14 in the
solvent leg, each running equilibration plus production, times the number of repeats.
`screening` brings the same system down to 134 ns. For the other protocols see
[RBFE](rbfe.md#choosing-the-edges), [SepTop](septop.md#cost-and-gpu-packing) and
[plain MD](md.md#cost).
