# Execution

```bash
openfe-api submit ./my_campaign --profiles profiles.yaml [--dry-run] [--only NAME] [--force]
```

## One process per repeat

OpenFE's own `protocol_repeats` runs every repeat inside a single process. openfe-api plans
with one repeat per transformation and runs each repeat as its own `openfe quickrun`
process instead, so repeats can be spread across GPUs, packed onto one GPU, and resumed
individually after a wall-time limit.

## Execution profiles

Where the work runs is described by a profile, kept separate from the request so the same
campaign is portable between machines. Copy `examples/profiles.yaml` and edit it: partition
names, queue names, wall times and environment activation all differ between clusters.

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

Alchemical windows rarely saturate a modern GPU. `jobs_per_gpu` above one runs that many
repeats against a single device through the Multi-Process Service, which raises throughput
substantially.

Repeats are packed into groups of `jobs_per_gpu`, filling every slot even when that means a
group spans two ligands, so a ligand with three repeats does not waste two slots of a
five-slot GPU. Tasks are ordered by run and then repeat, so a ligand's repeats stay together
wherever a group is large enough.

!!! warning "One repeat is not one GPU context"
    A multistate leg does not hold a single context. Minimization cannot run under a
    barostat, so openmmtools rebuilds each replica's state without one, and the rebuilt
    state loses the alchemical state that makes replicas share a context. OpenFE's context
    cache is unbounded, so **a leg leaves roughly one context per lambda window alive** for
    the rest of the run.

    Measured on a 36k-particle ABFE complex system:

    | | contexts | memory |
    |---|---|---|
    | propagation only | 1 | ~270 MiB |
    | after minimization, 30 windows | ~31 | ~8.4 GB |
    | two repeats sharing a GPU | ~62 | ~16.8 GB |

    Two repeats therefore exhaust a 16 GB card, and 62 contexts also exceeds the 48 client
    contexts MPS serves per device, so sharing fails on large cards too. Both were seen in
    testing, reported as `No compatible CUDA device is available` part way into the complex
    leg.

    `submit` warns when a profile asks for this, counting the windows the campaign's own
    protocol runs. What that means per protocol:

    | Protocol | Windows per process | Measured |
    |---|---|---|
    | ABFE | 30 complex, 14 solvent | two repeats exhaust a 16 GB card and have failed |
    | RBFE | 11 per phase | 2 and 3 per GPU completed on an L40S; a complex phase peaks at 6.9 GB |
    | SepTop | 27 solvent then 19 complex, one process | 2 per GPU completed on an L40S at 26.8 GB |

    The count is per phase, not per process: SepTop's two phases run in one process, and GPU
    memory was measured falling from 13.6 GB to 435 MiB at the boundary between them, so the
    cache is released when a phase's units finish.

    **Use `jobs_per_gpu: 1` for ABFE.** That is the one case where packing has actually been
    seen to fail. RBFE and SepTop both pack successfully on a 46 GB card, SepTop asking for 54
    contexts and getting them, so the documented 48-context ceiling is not the binding
    constraint there; memory is, at roughly 13.5 GB per SepTop job and 6.9 GB per RBFE complex
    phase. Those numbers are for one card, so measure your own: submit a single repeat with
    `jobs_per_gpu` raised, watch `nvidia-smi` through its minimization, and read the peak.

    The underlying behavior is an OpenFE/openmmtools issue, not something openfe-api can
    fix; a bounded energy context cache upstream would remove it.

## Slurm

`submit` writes one batch script per cost class in the campaign directory and submits each as
its own job array, one element per group. `--dry-run` writes the scripts without submitting,
which is worth doing once so you can read what will run.

A Slurm array shares a single wall-time limit, so transformations that need very different
amounts of time cannot share one. A charge-corrected RBFE edge runs 22 lambda windows for 20 ns
each, roughly four times a neutral edge, so it goes in a `charge` array of its own while the
rest go in a `standard` array. Give each class its own limit on the profile:

```yaml
slurm:
  time: 1-00:00:00
  time_by_class:
    charge: 4-00:00:00
```

A class with no entry falls back to `time`. A [relaxation](md.md#relaxation) queues in its own
`relax` class for the same reason: it is far shorter than the campaign it precedes.

The script stops the MPS daemon from a shell trap, so it is cleaned up even when a repeat
fails or the job is killed at its wall-time limit.

## Local

The local backend runs repeats on the current machine, spread across the GPUs it detects
from `CUDA_VISIBLE_DEVICES` or `nvidia-smi`. Each repeat writes `quickrun.log` in its own
working directory.

## Stopping a local run

Ctrl+C stops the whole batch in one press: running repeats are stopped, nothing further is
started, and the command exits with status 130. Stopped repeats are reported as
**interrupted**, not failed, and the run's state says to resubmit — their resume caches are
kept, so the next `submit` continues them.

## Resuming

Rerunning `submit` after an interruption picks up where it stopped:

- repeats that finished successfully are **skipped**
- repeats that **failed** start again from the beginning, since a failed run leaves a result
  file behind but nothing usable in it
- repeats with a `quickrun_cache` directory are **resumed**
- everything else starts fresh

In the batch script the decision is made at runtime, per repeat, so a requeued job continues
rather than starting over. `--force` reruns repeats that already finished.
