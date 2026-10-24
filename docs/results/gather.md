# Gathering results

```bash
openfe-api gather ./my_campaign [--only NAME] [--quiet]
```

!!! note "Not `openfe gather`"
    This is openfe-api's own command. It reads the `results.json` files the campaign's
    repeats wrote, combines them per run, applies the quality checks below, updates the
    manifest, and writes tables into `results/`. It does not call OpenFE's CLI, and OpenFE's
    `gather` is not run anywhere in this package. Where a number is defined the same way
    OpenFE defines it, this page says so.

Only the result files are read, never a trajectory, so gathering is cheap enough to run
anywhere.

| Protocol | Output | Table |
|---|---|---|
| ABFE | absolute ΔG per ligand | `results/results.tsv` |
| [RBFE](../protocols/rbfe.md#results), [SepTop](../protocols/septop.md#results) | ΔΔG per edge, plus fitted ligand values | `edges.tsv`, `ligands.tsv` |
| [Plain MD](../protocols/md.md#results) | the files each repeat wrote | `results/simulations.tsv` |

Against a remote service, every table here is downloadable; see
[getting files out](../running/service.md#getting-files-out).

## Binding free energies

`gather` prints one row per ligand:

| Ligand | dG (kcal/mol) | uncertainty | from | repeats | quality |
|---|---:|---:|---|---:|---|
| `ligand_1` | -9.23 | 0.91 | `std` | 3 | fail |

`from` names where the uncertainty came from: with several repeats it is their spread
(`std`), with one it falls back to the combined MBAR error of the two legs (`mbar`), which is
how `openfe gather-abfe` defines it.

## Quality checks

Each ligand is checked against OpenFE's own guidance:

| Check | Criterion |
|---|---|
| `mbar_overlap` | Overlap between neighboring lambda states at least **0.03** |
| `replica_exchange` | Neighboring states actually exchange, connecting the end states |
| `forward_reverse` | Forward and reverse estimates agree within their combined error, over the second half of the data |
| `repeat_spread` | Spread between repeats within 1 kcal/mol |
| `failed_repeats` | Every repeat produced a result |

The first three thresholds come from OpenFE's protocol documentation; the repeat spread is
this project's own rule of thumb. A check whose data is absent reports **unknown**, never a
pass:

```text
fail ligand_1 mbar_overlap: smallest neighboring overlap 0.002 (want >= 0.03)
                            in the complex leg of repeat repeat3
```

`forward_reverse` is judged over the second half of the data, and the gap is reported both
in kcal/mol and in multiples of the error.

A repeat that crashed, or never wrote an estimate, is reported as unusable and excluded from
the average rather than quietly dropped.
