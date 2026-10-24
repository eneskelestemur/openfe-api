---
name: run-campaign
description: Run an OpenFE free energy campaign end to end with the openfe-api CLI — validate, create, prep, plan, submit, gather, analyze — including which stage to resume from, how execution profiles work, and what each failure means. Use this whenever the task involves running, resuming, queueing, monitoring or debugging an openfe-api campaign, a binding free energy calculation (ABFE, RBFE, SepTop) or a plain MD run on a cluster, even when the user names only one stage ("plan this", "why did submit fail", "is it done yet").
---

# Running a campaign

openfe-api turns a request document into a campaign directory and walks it through fixed
stages. The campaign directory is the whole state: every stage reads it, writes to it, and
can run on a different machine from the one before.

## The flow

```bash
openfe-api validate request.yaml                              # no files touched
openfe-api create   request.yaml ./campaign
openfe-api relax    ./campaign --profiles profiles.yaml       # only if relax.enabled
openfe-api prep     ./campaign
openfe-api plan     ./campaign -p 8
openfe-api submit   ./campaign --profiles profiles.yaml --dry-run
openfe-api submit   ./campaign --profiles profiles.yaml
openfe-api status   ./campaign
openfe-api gather   ./campaign
openfe-api analyze  ./campaign --measured affinities.csv
```

Stop after `validate` if you are only checking a request. Stop after `plan` if the user
wants the cost before committing GPU time — `plan` prints the total nanoseconds the settings
imply, and that is the moment to confirm before spending an allocation.

Never skip `--dry-run` on the first `submit` of an unfamiliar profile. It writes the batch
script without queueing anything, and reading it catches a wrong partition or a missing
account before the scheduler does.

## Before running anything, read the state

`openfe-api status ./campaign` prints every run and its state: `pending`, `prepared`,
`planned`, `submitted`, `running`, `done` or `failed`. Each stage is idempotent and refuses
to redo finished work, so the right move is almost always to run the next stage rather than
to start over.

| State | Next |
|---|---|
| `pending` | `prep` |
| `prepared` | `plan` |
| `planned` | `submit` |
| `submitted` or `running` | wait; `status` again, or look in `logs/` |
| `failed` | read the reason in `manifest.json`, fix the input, rerun that stage |
| `done` | `gather`, then `analyze` |

Rerunning `submit` after an interruption skips finished repeats, resumes those with a
`quickrun_cache`, and restarts failed ones. `--force` reruns repeats that already finished;
only use it when the user explicitly wants the work redone.

## Execution profiles

`--profiles` is required for any stage that queues work (`relax`, `submit`, and
`analyze --submit`). The profile holds everything site-specific: partition, QoS, account,
wall time, GPU request, `jobs_per_gpu`. The request never holds these — if a request names
`backend` or `jobs_per_gpu` under `execution`, that is the error, and the fix is to move it
to the profile.

Copy `examples/profiles.yaml` and edit it for the site. Two rules worth knowing before
setting `jobs_per_gpu`:

- ABFE needs `jobs_per_gpu: 1`. Two repeats exhaust a 16 GB card.
- RBFE packs 2–3 per 46 GB card, SepTop 2, plain MD 3 or more.

`docs/running/execution.md` has the measured numbers behind those.

## Which protocol

The request's `protocol` field decides the shape of everything downstream. If the user has
not chosen, read `docs/protocols/index.md` — it compares the four on cost, charge handling
and whether an atom mapping is needed. Writing the request itself is the `write-request`
skill.

`--only NAME` restricts a stage to named runs, and is **rejected** for RBFE and SepTop,
whose runs are network edges that do not exist until `plan` has run.

## When a stage fails

Exit codes: `0` success, `1` the command failed or a run inside it failed, `130` a local
`submit` was interrupted.

Error messages name the file, the residue and the rule that was broken — read the message
before investigating anything else, because this package is built to fail at the earliest
stage that can detect a problem. In particular:

- A chemistry failure during `prep` is an input problem; see the `write-request` skill.
- A run that failed during `submit` leaves its log in `runs/<run>/repeatN/quickrun.log` and
  the scheduler's output under `logs/`.
- `gather` exiting non-zero means nothing usable was produced, not that gathering broke.

Do not edit anything inside the campaign directory by hand to work around a failure. The
manifest is the source of truth for both the CLI and the service, and hand-editing it is how
a campaign becomes unresumable.

## Long work belongs on a compute node

Preparation and planning are seconds to minutes and are fine on a login node. Simulation is
always queued. Trajectory analysis reads tens of gigabytes, so it takes `--submit`:

```bash
openfe-api analyze ./campaign --trajectories --workers 16 \
    --submit --profiles profiles.yaml --profile cpu
```

## Reference

| Question | File |
|---|---|
| What each command takes | `docs/reference/cli.md` |
| What the campaign directory holds | `docs/start/campaign.md` |
| Profiles, Slurm, GPU packing, resuming | `docs/running/execution.md` |
| Worked requests to copy | `examples/*.yaml` |
