# Results

```bash
openfe-api gather ./my_campaign [--only NAME] [--quiet]
```

Everything needed is inside the result files that `openfe quickrun` writes, so gathering
reads no trajectories and is cheap enough to run anywhere.

What `gather` produces depends on the protocol:

| Protocol | Output | Table |
|---|---|---|
| ABFE | absolute ΔG per ligand, below | `results/results.tsv` |
| [RBFE](rbfe.md#results), [SepTop](septop.md#results) | ΔΔG per edge, plus fitted ligand values | `edges.tsv`, `ligands.tsv` |
| [Plain MD](md.md#results) | the files each repeat wrote | `results/simulations.tsv` |

The rest of this page covers ABFE. The quality checks below apply to every protocol that
produces a free energy. Working against a remote service, every table and artifact here is
also downloadable; see [getting files out](service.md#getting-files-out).

## Binding free energies

`gather` prints one row per ligand:

| Ligand | dG (kcal/mol) | uncertainty | from | repeats | quality |
|---|---:|---:|---|---:|---|
| `ligand_1` | -9.23 | 0.91 | `std` | 3 | fail |

`from` names where the uncertainty came from. With several repeats it is their spread
(`std`); with one it falls back to the combined MBAR error of the two legs (`mbar`), matching
`openfe gather-abfe`. The table is written to `results/results.tsv`.

## Quality checks

Each ligand is checked against OpenFE's own guidance:

| Check | Criterion |
|---|---|
| `mbar_overlap` | Overlap between neighboring lambda states at least **0.03** |
| `replica_exchange` | Neighboring states actually exchange, connecting the end states |
| `forward_reverse` | Forward and reverse estimates agree within their combined error, judged over the second half of the data |
| `repeat_spread` | Spread between repeats within 1 kcal/mol |
| `failed_repeats` | Every repeat produced a result |

The first three thresholds come from OpenFE's protocol documentation. The repeat spread
threshold is this project's own, and is a rule of thumb rather than a standard.

A check whose data is absent is reported as **unknown**, never as a pass:

```text
fail ligand_1 mbar_overlap: smallest neighboring overlap 0.002 (want >= 0.03)
                            in the complex leg of repeat repeat3
```

Convergence is judged on the second half of the data only: the earliest fractions disagree
even for a converged run, and the final point is identical by construction since both
directions then use everything. The gap is reported in kcal/mol as well as in multiples of the
error, because MBAR's analytical errors are small enough that a modest gap is many sigma.

A failing check does not mean the number is wrong, but it does mean it should not be trusted
without looking further. Poor overlap usually calls for more lambda windows; a wide spread
between repeats calls for longer sampling.

## Failed repeats

A repeat whose simulation crashed, or which never wrote an estimate, is reported as unusable
and excluded from the average rather than quietly dropped.
