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
| `metrics.json` | Pearson, Spearman, Kendall, MUE and RMSE, and the package versions behind everything else here |
| `plots/` | The overlap, exchange and convergence plots OpenFE wrote, linked from `runs/` |
| `stability.png`, `stability.tsv` | Ligand drift per run, with `--trajectories` |
| `trajectory.tsv` | Protein RMSD, ligand RMSD and drift per frame, with `--trajectories` |
| `interactions.png`, `interactions.tsv` | How often each pocket residue was contacted, with `--trajectories` |
| `scenes/` | A representative pose per run, to open in PyMOL or a browser, with `--trajectories` |

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

A column whose values are all positive is called out, and so is a correlation that comes out
negative. A pIC50 rises where a free energy falls, so a column of affinities used directly
reads as the predictions getting every ligand backwards.

A ligand that has no measurement is not dropped. It appears beside the correlation figure at
its predicted value, so you can see where it is expected to fall. A measured ligand that the
campaign does not contain is reported as a warning.

## What each protocol gets

ABFE predicts absolute binding free energies, so they are plotted as they are. RBFE and
SepTop fit each ligand against the network mean, so their values carry an arbitrary offset:
the predictions are shifted onto the mean of the measurements, and the shift is recorded in
`metrics.json`. Only the predictions move, so the measured axis keeps real free energies.
Correlation and ranking are unaffected either way.

`predictions.tsv` holds the values as the fit produced them and `correlation.tsv` holds them
after that shift, so the same ligand reads differently in the two files by exactly the shift.

Plain MD produces no free energy, so `analyze` refuses it. Metrics over its trajectories are
a later milestone.

## Trajectory metrics

`--trajectories` measures how far each ligand moved during its run: protein backbone RMSD,
symmetry-corrected ligand RMSD, and ligand center-of-mass drift. It reads the trajectories,
so it takes minutes where the rest of the analysis takes seconds, and is off by default.

An alchemical run is read at its fully interacting end state; plain MD is read whole. Each
repeat is measured separately and then aggregated per run, so disagreement between repeats
stays visible. A repeat that cannot be measured — too short to have two frames, or missing
its topology — is reported as a warning and the rest carry on.

Drift is movement relative to the protein, so it is left out of a solvent phase and of a
ligand simulated on its own: aligned to itself, a ligand reports no drift whatever it does.
An apo system reports protein RMSD and nothing else.

## Pocket interactions

`--trajectories` also records which residues the ligand touched, frame by frame: hydrophobic
contacts, hydrogen bonds either way round, pi stacking, salt bridges and cation-pi. Each is
reported as an occupancy, the fraction of frames it was present in, averaged over repeats.

The heatmap puts every run in one figure, so a substitution that trades one hydrogen bond
for another is visible without flicking between images. Residues are kept when something
occupies them at least 10% of the time in some run, and van der Waals contacts are left out
of the figure: they are present nearly everywhere and say little. `interactions.tsv` holds
everything, including what the figure drops.

Only residues within 8 A of the ligand are considered, worked out once from the first frame.

## Representative poses

One frame per run is written out for looking at: the frame whose ligand sits closest to the
median position it held over the run, which is the pose it spent most of its time near
rather than an average of poses that never existed.

Each run gets three files in `scenes/`:

| File | What it is |
|---|---|
| `<run>.pdb` | The frame itself, protein and ligand |
| `<run>.pml` | A PyMOL script: cartoon protein, sticks for the ligand, pocket residues shown |
| `<run>.html` | A standalone page that draws the same scene with py3Dmol, for a browser |

```bash
pymol analysis/scenes/lig_1.pml
```

Nothing here needs PyMOL installed. The `.pml` file is a text script, written whether or
not you have it.

## Running it on a cluster

Measuring the trajectories of a production campaign reads tens of gigabytes, so it is work
for a compute node rather than a login node. `--submit` sends the analysis to Slurm, and
`--workers` says how many trajectories to measure at once:

```bash
openfe-api analyze ./my_campaign --measured affinities.csv \
    --trajectories --workers 16 --submit --profiles profiles.yaml --profile cpu
```

Measuring trajectories wants cores rather than a GPU, so `--profile` names the profile the
analysis job runs under; the campaign's own, which it falls back to, is usually the wrong
shape. The job asks for at least one core per worker, and everything else about the profile
-- the partition, QoS, account and memory -- is used exactly as written, so give it a
`time_by_class` entry named `analysis` if the analysis needs a different wall time from the
simulations.

`--dry-run` writes the batch script without submitting it, so you can read it first.

A worker takes a couple of seconds to start, so there is nothing to gain on a short test
run; the win is on production-length trajectories, where each one takes minutes.

## Figures

`--dark` draws for a dark background instead of a light one. The tolerance band on the
correlation figure marks agreement within 1 kcal/mol.

With no measurements at all you still get the ranking figure, which is the useful view when
a campaign is finished but the assay is not.
