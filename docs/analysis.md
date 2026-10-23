# Analysis

A finished campaign turns into figures and tables with one command. It is read-only and
repeatable: nothing about the campaign changes, so run it again whenever new measurements
arrive.

```bash
openfe-api analyze ./my_campaign --measured affinities.csv
```

Everything lands in `analysis/` inside the campaign.

| File | What it holds |
|---|---|
| `ranking.png` | Ligands ordered by predicted ΔG, with their uncertainties |
| `predictions.tsv` | The same numbers |
| `correlation.png` | Predicted against measured, when measurements were given |
| `correlation.tsv` | Every point behind that figure |
| `metrics.json` | Pearson, Spearman, Kendall, MUE and RMSE |

Every figure has its numbers written beside it, so you can re-plot without re-running.

## Measured values

A CSV or TSV with a ligand name column (`name` or `ligand`) and a free energy column in
kcal/mol (`dg`, `dg (kcal/mol)`, `experimental` or `experimental_dg`), plus an optional
uncertainty. Column names are matched ignoring case and spacing.

```csv
name,dg,uncertainty
ligand_1,-9.2,0.3
ligand_2,-8.1,0.3
```

**Values are used exactly as given.** Turning an IC50 or a Kd into a free energy rests on
assumptions about the assay that this project will not make for you, so the conversion, and
the responsibility for it, stays with you.

A ligand that has no measurement is not dropped. It appears beside the correlation figure at
its predicted value, so you can see where it is expected to fall. A measured ligand that the
campaign does not contain is reported as a warning.

## What each protocol gets

ABFE predicts absolute binding free energies, so they are plotted as they are. RBFE and
SepTop fit each ligand against the network mean, so their values carry an arbitrary offset:
the predictions are shifted onto the mean of the measurements, and the shift is recorded in
`metrics.json`. Only the predictions move, so the measured axis keeps real free energies.
Correlation and ranking are unaffected either way.

Plain MD produces no free energy, so `analyze` refuses it. Metrics over its trajectories are
a later milestone.

## Figures

`--dark` draws for a dark background instead of a light one. The tolerance band on the
correlation figure marks agreement within 1 kcal/mol.

With no measurements at all you still get the ranking figure, which is the useful view when
a campaign is finished but the assay is not.
