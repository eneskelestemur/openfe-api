# Absolute binding free energy

Each complex listed in the request is an independent run, prepared and simulated on its own,
so a failure costs you that ligand and nothing else.

## The request

```yaml
protocol: abfe
name: my_abfe_campaign

complexes:
  - name: ligand_1              # one run each
    structure: ./model.cif      # combined file, or per-component paths below
    protein:
      chains: [A, B]            # omit to keep every chain
      ph: 7.4
      keep_waters: false
    ligand:
      selector: {chain: E, resname: LIG_E}
      smiles: "c1ccc(cc1)C(=O)NC2CCCCC2"
    extra_ligand_copies: drop   # drop | keep; required for a combined file
    cofactors:
      - name: NAD_C
        selector: {chain: C}
        smiles: "NC(=O)c1ccc[n+](...)c1"

neutralize_ligands: false

settings:
  preset: default
execution:
  profile: cluster_gpu
  repeats: 3
```

Each entry under `complexes` is prepared into `prepared/<name>/`. The rules every protocol
shares — paths, SMILES, stereochemistry, selectors — are in the
[input contract](../running/inputs.md).

## Rules of its own

**The alchemical ligand must be neutral.** `AbsoluteBindingProtocol` refuses a charged one,
and preparation checks it first, so the failure arrives in seconds rather than after a GPU
has started. [`neutralize_ligands`](../running/inputs.md#charges) can remove the charge where
protonation allows. Cofactors may carry any charge.

**Three or more fused aromatic rings are refused.** The complex phase picks Boresch restraint
atoms automatically, and OpenFE 1.12's search fails on anthracene, phenanthrene, carbazoles,
acridines and pyrrolo-quinoxalines. Planning calls OpenFE's own function up front, so the
complex is refused with an explanation instead of failing mid-run.

## Cost

Per complex, per repeat, with OpenFE's defaults:

| Phase | Windows | Simulation |
|---|---|---|
| solvent | 14 | 155 ns |
| complex | 30 | 336 ns |
| total | | **491 ns** |

The `screening` preset brings the same system down to 134 ns.

Use `jobs_per_gpu: 1`: a complex leg leaves roughly one GPU context per window alive, so two
ABFE repeats exhaust a 16 GB card. See
[sharing a GPU](../running/execution.md#sharing-a-gpu-with-cuda-mps).

## Results

`openfe-api gather` writes one row per ligand to `results/results.tsv` and checks each
against OpenFE's convergence criteria. See [gathering results](../results/gather.md).
