---
name: read-results
description: Read and interpret what an openfe-api campaign produced — gathered free energies, per-edge and fitted per-ligand values, convergence and cycle-closure checks, correlation against measured affinities, trajectory stability and pocket interactions. Use this whenever the task involves interpreting openfe-api output, comparing predictions against experiment, deciding whether a free energy result can be trusted, or explaining a results table, a quality check, a metrics.json or an analysis figure.
---

# Reading results

Two commands produce everything, and they are different jobs:

- `openfe-api gather` reads the `results.json` files the repeats wrote, combines them,
  applies quality checks, and writes tables into `results/`. It is **not** `openfe gather`
  — this package never calls OpenFE's CLI except `openfe quickrun`.
- `openfe-api analyze` reads what `gather` left and writes figures, correlation and
  trajectory metrics into `analysis/`. It changes no run state and can be rerun freely.

## What gather writes

| Protocol | Table | Holds |
|---|---|---|
| ABFE | `results/results.tsv` | absolute ΔG per ligand |
| RBFE, SepTop | `results/edges.tsv`, `results/ligands.tsv` | ΔΔG per edge, and per-ligand values fitted over the network |
| Plain MD | `results/simulations.tsv` | which repeats finished and where their files are |

**A network's per-ligand values are relative.** The fit is centered on the network mean, so
differences between ligands are meaningful and a single value on its own is not. Never
present a number from `ligands.tsv` as an absolute binding free energy.

The `from` column says where an uncertainty came from: `std` is the spread across repeats,
`mbar` is the MBAR error used when there is only one repeat.

## Judging whether to trust it

`gather` checks each result against OpenFE's own criteria: `mbar_overlap` (≥ 0.03 between
neighboring lambda states), `replica_exchange`, `forward_reverse`, `repeat_spread` (within
1 kcal/mol, this project's own rule of thumb) and `failed_repeats`.

A check whose data is missing reports `unknown`, never a pass — treat `unknown` as "not
checked", not as "fine". A failing check does not make the number wrong, but it should not
be reported without the caveat.

For a network there is also **cycle closure**, which needs no experimental data: what the
free energies around each cycle actually sum to. Over 1 kcal/mol is a failure. A network
planned without redundancy has no cycles and reports `unknown`, which is worth saying out
loud when someone asks how good a network is.

A failed edge can disconnect part of the network; gathering names the ligands that lost
their path and leaves them out of the fit. Check for those before reporting coverage.

## Comparing against measurement

```bash
openfe-api analyze ./campaign --measured affinities.csv
```

The table needs a name column (`name` or `ligand`) and a free energy column in **kcal/mol**
(`dg`, `dg (kcal/mol)`, `experimental`, `experimental_dg`), plus an optional uncertainty.

**Values are used exactly as given, and no affinity is converted.** A pIC50 or pKi rises
where a free energy falls, so passing one straight in inverts the whole comparison. If the
user's numbers are affinities, convert them deliberately and say what assumptions the
conversion made — or tell them the analysis will read them backwards. Two warnings exist for
exactly this: an all-positive measured column, and a negative correlation. Both mean "check
the column before believing the plot".

For RBFE and SepTop the predictions are shifted onto the measured mean, and the shift is
recorded in `metrics.json`. Only the predictions move, so the measured axis keeps real
values. This is why `predictions.tsv` and `correlation.tsv` differ for the same ligand by
exactly that shift.

`metrics.json` also records the package versions behind every file in the directory, and is
written whether or not anything was measured.

## Trajectory metrics

`--trajectories` adds protein backbone RMSD, symmetry-corrected ligand RMSD, ligand
center-of-mass drift, pocket interaction occupancies, and one representative pose per run
under `analysis/scenes/`. It reads every trajectory, so on a production campaign submit it
rather than running it inline:

```bash
openfe-api analyze ./campaign --trajectories --workers 16 \
    --submit --profiles profiles.yaml --profile cpu
```

A repeat too short to measure is reported as a warning and the rest carry on, so check the
warnings before concluding that a run was stable.

One caveat worth repeating when showing these: the plots OpenFE itself wrote into
`analysis/plots/` are computed without alignment, so their drift numbers are much larger
than the aligned ones in `stability.tsv`. Use `stability.tsv`.

## Reference

| Question | File |
|---|---|
| Every table and check | `docs/results/gather.md` |
| Every analysis output and flag | `docs/results/analysis.md` |
| What a protocol's results mean | `docs/protocols/<protocol>.md` |
