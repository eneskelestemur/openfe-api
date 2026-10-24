# Analysis

```bash
openfe-api analyze ./my_campaign --measured affinities.csv
```

Read-only and repeatable: no run state changes, so it can be rerun whenever new measurements
arrive. Everything lands in `analysis/`.

| File | What it holds |
|---|---|
| `ranking.png`, `predictions.tsv` | Ligands ordered by predicted ΔG, with uncertainties |
| `correlation.png`, `correlation.tsv` | Predicted against measured, when measurements were given |
| `metrics.json` | Pearson, Spearman, Kendall, MUE, RMSE, and the package versions behind everything here |
| `plots/` | The overlap, exchange and convergence plots OpenFE wrote, linked from `runs/` |
| `stability.png`, `stability.tsv`, `trajectory.tsv` | Ligand and protein movement, with `--trajectories` |
| `interactions.png`, `interactions.tsv` | Pocket contact occupancy, with `--trajectories` |
| `scenes/` | A representative pose per run, with `--trajectories` |

Every figure has its numbers written beside it, so you can re-plot without re-running.
`--dark` draws for a dark background.

## Measured values

A CSV or TSV with a ligand name column (`name` or `ligand`) and a free energy column in
kcal/mol (`dg`, `dg (kcal/mol)`, `experimental` or `experimental_dg`), plus an optional
uncertainty. Column names are matched ignoring case and spacing.

```csv
name,dg,uncertainty
ligand_1,-9.2,0.3
ligand_2,-8.1,0.3
```

**Values are used exactly as given.** No affinity is converted for you, since the conversion
rests on assumptions about the assay. A column whose values are all positive is called out,
and so is a correlation that comes out negative — both are what passing a pIC50 straight in
looks like.

A ligand with no measurement is not dropped — it appears beside the correlation figure at its
predicted value. A measured ligand the campaign does not contain is reported as a warning.

## What each protocol gets

ABFE values are plotted as they are. RBFE and SepTop values carry the network's arbitrary
offset, so the predictions are shifted onto the mean of the measurements and the shift is
recorded in `metrics.json`. Only the predictions move, so the measured axis keeps real free
energies, and correlation and ranking are unaffected either way.

`predictions.tsv` holds the values as the fit produced them and `correlation.tsv` holds them
after that shift, so the same ligand differs between the two files by exactly the shift.

Plain MD produces no free energy, so `analyze` refuses it unless `--trajectories` is given.

## Trajectory metrics

`--trajectories` adds protein backbone RMSD, symmetry-corrected ligand RMSD and ligand
center-of-mass drift. It reads the trajectories, so it takes minutes where the rest takes
seconds, and is off by default.

An alchemical run is read at its fully interacting end state; plain MD is read whole. Each
repeat is measured separately and then aggregated per run, so disagreement between repeats
stays visible. A repeat that cannot be measured — too short, or missing its topology — is
reported as a warning and the rest carry on.

Drift is measured against the protein, so it is left out of a solvent phase and of a ligand
simulated alone. An apo system reports protein RMSD and nothing else.

## Pocket interactions

The same flag fingerprints each frame with ProLIF: hydrophobic contacts, hydrogen bonds
either way round, pi stacking, salt bridges and cation-pi. Each is reported as an occupancy,
the fraction of frames it was present in, averaged over repeats.

The pocket is the residues within 8 Å of the ligand, taken once from the first frame. One
heatmap covers every run; it keeps a residue when something occupies it at least 10% of the
time in some run, and leaves out van der Waals contacts, which fire nearly everywhere.
`interactions.tsv` holds everything, including what the figure drops.

## Representative poses

One frame per run is written out: the one whose ligand sits closest to the median position
it held over the run.

| File | What it is |
|---|---|
| `scenes/<run>.pdb` | The frame itself, protein and ligand |
| `scenes/<run>.pml` | A PyMOL script: cartoon protein, sticks for the ligand, pocket residues shown |
| `scenes/<run>.html` | A standalone page drawing the same scene with py3Dmol |

```bash
pymol analysis/scenes/lig_1.pml
```

Nothing here needs PyMOL installed; the `.pml` file is a text script either way.

## Running it on a cluster

Measuring a production campaign's trajectories reads tens of gigabytes, so it belongs on a
compute node. `--submit` sends the analysis to Slurm and `--workers` says how many
trajectories to measure at once:

```bash
openfe-api analyze ./my_campaign --measured affinities.csv \
    --trajectories --workers 16 --submit --profiles profiles.yaml --profile cpu
```

`--profile` names the profile the job runs under, since the campaign's own is a GPU profile
and this is CPU work. The job asks for at least one core per worker; everything else —
partition, QoS, account, memory — is used as written, so give the profile a `time_by_class`
entry named `analysis` if the analysis needs its own wall time. `--dry-run` writes the batch
script without submitting it.

A worker takes a couple of seconds to start, so there is nothing to gain on a short run.
