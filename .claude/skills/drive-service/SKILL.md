---
name: drive-service
description: Drive a running openfe-api HTTP service — create a campaign by uploading input files, start prep, plan and submit, poll for progress, and pull results or whole campaigns back over HTTP. Use this whenever the task involves the openfe-api service or container rather than the local CLI — a base URL or port is given, the structures are on a different machine from the cluster, the user mentions the OpenAPI schema or /docs, or an agent needs to run free energy calculations remotely without filesystem access.
---

# Driving the service

The service is the same library behind HTTP, so everything the CLI does to a campaign
directory happens here too. The difference is that a client elsewhere cannot see that
filesystem, which shapes how inputs go in and results come out.

`GET /docs` serves the OpenAPI schema generated from the same models the CLI validates
against. When a field here is unclear, read the schema rather than guessing — it is always
current.

## Get the inputs in

If the structures already sit on the service's filesystem, `POST /campaigns` takes the
request as JSON with ordinary paths. Otherwise upload them with the request:

```bash
curl --fail-with-body http://HOST:8000/campaigns/upload \
    --form-string "campaign=$(cat campaign.json)" \
    --form 'files=@bound.cif' \
    --form 'files=@cofactor.sdf'
```

Every path inside the uploaded request must be a **bare filename** matching one of the
`files` parts — no directories. The campaign keeps its own copy under `inputs/`. An existing
campaign is never overwritten, and a failed upload leaves nothing behind. Chemistry is
validated at `prep`, not at upload, so a 200 here does not mean the request is sound.

Writing the request itself is the `write-request` skill.

## Run the stages

```bash
curl -X POST http://HOST:8000/campaigns/my_run/prep
curl -X POST http://HOST:8000/campaigns/my_run/plan
curl -X POST http://HOST:8000/campaigns/my_run/submit
```

These three return **202 Accepted** and run in the background; the response body is not the
result. One campaign runs one operation at a time, and a duplicate request while it is busy
is ignored and logged, so firing `prep` twice is harmless but does nothing.

Poll `GET /campaigns/{name}` for both the state of every run and a `last_operation` record:

```json
{"operation": "plan", "status": "failed", "started_at": "...", "finished_at": "...",
 "message": "planned 0, failed 1, 0 ns total"}
```

A background failure is visible here and nowhere else, so check `last_operation.status`
before moving to the next stage — a stage that failed still returns 202 for the next one.
Poll on a timescale that matches the work: seconds for `prep`, minutes for `plan`, and for
`submit` expect hours or days, so poll slowly and report rather than block.

## Read the results

Each protocol has its own endpoint, and asking the wrong one returns `409` naming the right
one:

| Protocol | Endpoint |
|---|---|
| ABFE | `GET /campaigns/{name}/results` |
| RBFE, SepTop | `GET /campaigns/{name}/network` |
| Plain MD | `GET /campaigns/{name}/simulations` |

`POST /campaigns/{name}/analysis` writes the figures and tables and returns the metrics. It
answers in the request rather than in the background, so with `"trajectories": true` it can
take minutes to hours: raise `"workers"`, set a client timeout to match, and for a
production campaign prefer `openfe-api analyze --submit` on the cluster instead.

What the numbers mean is the `read-results` skill.

## Get the files out

```bash
curl "http://HOST:8000/campaigns/my_run/files?group=results"     # list
curl -O http://HOST:8000/campaigns/my_run/files/results/results.tsv
curl -O http://HOST:8000/campaigns/my_run/archive                 # everything but runs/
```

Groups are `inputs`, `prepared`, `plans`, `relaxed`, `runs`, `results` and `logs`. Each
listed `path` goes straight into the download endpoint.

For anything large, pull files individually rather than as an archive: single files support
ranged requests so a dropped connection resumes, while a generated archive does not. Ask for
`group=runs` only when the trajectories are actually wanted — that is the gigabytes.

## Before pointing anyone at a URL

The service has **no authentication**. Anyone who can reach the port can read every
campaign, create campaigns and spend the allocation. Do not expose it on an open interface,
do not suggest binding `0.0.0.0` outside a container whose port is deliberately published,
and if the user is about to, say so.

`docs/running/service.md` has the full endpoint table, the configuration variables and the
container recipes.
