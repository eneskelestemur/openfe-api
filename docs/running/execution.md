# Execution

```bash
openfe-api submit ./my_campaign --profiles profiles.yaml [--dry-run] [--only NAME] [--force]
```

Each repeat runs as its own `openfe quickrun` process, rather than inside one process as
OpenFE's `protocol_repeats` would, so repeats can be spread across GPUs, packed onto one, and
resumed individually after a wall-time limit. `submit` writes the scripts and `--dry-run`
stops before they are sent.

## Execution profiles

A profile describes where the work runs, kept separate from the request so the same campaign
is portable. Copy `examples/profiles.yaml` and edit it.

```yaml
cluster_gpu:
  backend: slurm
  jobs_per_gpu: 3
  setup_commands:
    - source ~/.bashrc
    - conda activate openfe-api
  slurm:
    partition: gpu
    qos: gpu_access        # remove if your cluster does not use one
    time: 3-00:00:00       # quote a bare HH:MM:SS: YAML reads it as a base-60 number
    gres: gpu:1
    cpus_per_task: 12
    memory: 128G
    extra_directives:
      - --account=my_account
```

## Sharing a GPU with CUDA MPS

`jobs_per_gpu` above one runs that many repeats against a single device through the
Multi-Process Service. Repeats are packed into groups of that size, filling every slot even
when a group spans two ligands, so a ligand with three repeats does not waste two slots of a
five-slot GPU. Tasks are ordered by run and then repeat, so a ligand's repeats stay together
where a group is large enough.

!!! warning "One repeat is not one GPU context"
    **A leg leaves roughly one GPU context per lambda window alive** for the rest of the run,
    because OpenFE's context cache is unbounded and minimization rebuilds each replica's
    state into one of its own. On a 36k-particle ABFE complex: 1 context and ~270 MiB during
    propagation, ~31 contexts and ~8.4 GB after minimization.

What that means per protocol, measured:

| Protocol | Windows per phase | Result |
|---|---|---|
| ABFE | 30 complex, 14 solvent | **use `jobs_per_gpu: 1`** — two repeats exhaust a 16 GB card and exceed the 48 contexts MPS serves; both have failed in testing |
| RBFE | 11 per phase | 2 and 3 per GPU completed on a 46 GB L40S; a complex phase peaks at 6.9 GB |
| SepTop | 27 solvent, then 19 complex | 2 per GPU completed at 26.8 GB, asking for 54 contexts and getting them |
| Plain MD | 1 context total | 3 per card at 523 MiB between them |

The count is per phase, not per process: SepTop runs both phases in one process, and GPU
memory falls from 13.6 GB to 435 MiB at the boundary, so the cache is released when a phase
finishes. Memory, not the context ceiling, is the binding constraint — roughly 13.5 GB per
SepTop job and 6.9 GB per RBFE complex phase.

`submit` warns when a profile asks for more than the campaign's protocol has been seen to
take. Those numbers are for one card, so measure your own: submit a single repeat with
`jobs_per_gpu` raised, watch `nvidia-smi` through its minimization, and read the peak.

The behavior is upstream, in OpenFE and openmmtools, not something this package can fix.

## Slurm

One batch script per **cost class** is written in the campaign directory and submitted as its
own job array, one element per group. A Slurm array shares a single wall-time limit, so a
charge-corrected RBFE edge — roughly four times a neutral one — goes in a `charge` array of
its own rather than forcing every edge to book its limit. Give each class a limit:

```yaml
slurm:
  time: 1-00:00:00
  time_by_class:
    charge: 4-00:00:00
```

A class with no entry falls back to `time`. A [relaxation](../protocols/md.md#relaxation)
queues in its own `relax` class, and `analyze --submit` in an `analysis` class.

The script stops the MPS daemon from a shell trap, so it is cleaned up even when a repeat
fails or the job hits its wall-time limit.

## Local

The local backend runs repeats on the current machine, across the GPUs it detects from
`CUDA_VISIBLE_DEVICES` or `nvidia-smi`. Each repeat writes `quickrun.log` in its own working
directory.

Ctrl+C stops the whole batch in one press: running repeats are stopped, nothing further
starts, and the command exits 130. Those repeats are reported as **interrupted**, not failed,
and their resume caches are kept.

## Resuming

Rerunning `submit` picks up where it stopped:

- repeats that finished successfully are **skipped**
- repeats that **failed** start again from the beginning, since a failed run leaves a result
  file behind with nothing usable in it
- repeats with a `quickrun_cache` directory are **resumed**
- everything else starts fresh

In the batch script the decision is made at runtime, per repeat, so a requeued job continues
rather than starting over. `--force` reruns repeats that already finished.
