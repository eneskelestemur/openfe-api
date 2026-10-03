# Service and container

The same library is also an HTTP service, so a container can accept campaigns and submit
them to a cluster's scheduler.

```bash
uvicorn openfe_api.service:app --host 0.0.0.0 --port 8000
```

Interactive documentation is served at `/docs`, generated from the same models the CLI uses.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Service status and version |
| `GET` | `/campaigns` | List campaigns |
| `POST` | `/campaigns` | Create one from a request body (`?force=true` to overwrite) |
| `POST` | `/campaigns/upload` | Create one from uploaded input files and a JSON request |
| `GET` | `/campaigns/{name}` | Run states and the last operation |
| `POST` | `/campaigns/{name}/prep` | Prepare inputs |
| `POST` | `/campaigns/{name}/plan` | Charges and transformations |
| `POST` | `/campaigns/{name}/submit` | Run or queue repeats |
| `GET` | `/campaigns/{name}/results` | Absolute free energies with quality checks. ABFE only |
| `GET` | `/campaigns/{name}/network` | Edges, fitted ligand values and cycle closure. RBFE and SepTop |
| `GET` | `/campaigns/{name}/simulations` | What each repeat produced. Plain MD |
| `GET` | `/campaigns/{name}/files` | List the files the campaign holds |
| `GET` | `/campaigns/{name}/files/{path}` | Download one of them |
| `GET` | `/campaigns/{name}/archive` | Download the campaign as one tar |

The request body for `POST /campaigns` is the same structure as the YAML request, as JSON.

## Uploading inputs

`POST /campaigns` resolves input paths on the service's filesystem, which a client elsewhere
cannot write to. `POST /campaigns/upload` takes the files with the request instead: a `campaign`
form field holding the same JSON, and one `files` part per input.

```bash
curl --fail-with-body http://localhost:8000/campaigns/upload \
    --form-string "campaign=$(cat campaign.json)" \
    --form 'files=@bound.cif'
```

Every path in the request must be a bare filename matching an uploaded one, with a suffix of
`.cif`, `.mmcif`, `.pdb`, `.sdf`, `.mol` or `.mol2`. Combined structures, split protein and
ligand files and cofactors all work, as does any protocol. The campaign gets its own copy of
each file under `inputs/`, so it does not depend on where the upload came from.

An existing campaign is never overwritten, and a failed upload leaves no directory behind.
Chemistry is validated during `prep`, not here.

!!! warning "Set a body limit on your proxy too"
    `OPENFE_API_MAX_UPLOAD_BYTES` is enforced while the files are written, but the multipart
    parser has already received and spooled the request by then. A reverse proxy in front of
    the service should impose its own request-body limit, with room for multipart overhead.

## Getting files out

A client that cannot see the service's filesystem reads everything back over HTTP. Ask what
the campaign holds:

```bash
curl localhost:8000/campaigns/my_run/files?group=results
```

```json
[
  {
    "path": "results/results.tsv",
    "group": "results",
    "size": 412,
    "modified": "2026-10-03T09:12:44Z"
  }
]
```

The groups are `inputs`, `prepared`, `plans`, `relaxed`, `runs`, `results` and `logs`. Repeat
`group=` for several, or leave it off for everything. Each `path` goes straight into the
download endpoint:

```bash
curl -O localhost:8000/campaigns/my_run/files/results/results.tsv
```

A plain MD campaign's `/simulations` response reports its artifacts as these same paths, so
you can list the repeats and fetch their trajectories without knowing the layout.

### Everything at once

```bash
curl -O localhost:8000/campaigns/my_run/archive
```

That is one uncompressed tar, and it carries every group **except `runs`**. Simulation output
is the bulk of a campaign — a production run keeps gigabytes under `runs/` — so you ask for it
deliberately with `?group=runs`.

The tar is uncompressed on purpose. `.nc` and `.xtc` files are dense binary, so gzip reclaims
under a fifth of their size while compressing at roughly 20 MB/s, which is slower than most
links; on a 100 GB campaign that is over an hour of CPU spent to save very little. Leaving it
off also means the response can state its `Content-Length` up front, so your client shows real
progress and notices a truncated download. Pass `?compress=true` if you would rather have the
gzip anyway.

### Moving a lot of data

For anything large, pull the files individually rather than as an archive. Single files support
ranged requests, so a dropped connection resumes where it stopped, while a generated archive
cannot. Turn the listing into a URL list and hand it to a downloader that retries and
parallelizes:

```bash
BASE=localhost:8000/campaigns/my_run
curl -s "$BASE/files?group=runs" \
    | python -c 'import json,sys;[print(f["path"]) for f in json.load(sys.stdin)]' \
    | sed "s|^|$BASE/files/|" > urls.txt
aria2c -x8 -i urls.txt          # or: wget -c -i urls.txt
```

`curl -C -` resumes a single file the same way. If a campaign ever outgrows this, the answer is
object storage rather than a bigger archive endpoint: writing artifacts to S3 or MinIO and
handing back presigned URLs gives parallel, resumable transfer for free.

## Long operations

Preparation takes seconds, planning minutes, and a local run can take days — none of which
fits inside an HTTP request. Those three endpoints start the work in the background and
return `202 Accepted` immediately.

Progress is read back from `GET /campaigns/{name}`, which reports both the state of every
run and a `last_operation` record:

```json
{
  "operation": "plan",
  "status": "failed",
  "started_at": "2026-09-16T10:00:00Z",
  "finished_at": "2026-09-16T10:00:04Z",
  "message": "planned 0, failed 1, 0 ns total"
}
```

A failure inside a background task is therefore still visible. One campaign runs one
operation at a time; a duplicate request while it is busy is ignored and logged.

## Configuration

Every setting is an environment variable prefixed with `OPENFE_API_`. See `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `OPENFE_API_ROOT` | `campaigns` | Directory holding one subdirectory per campaign |
| `OPENFE_API_PROFILES_FILE` | unset | YAML file of execution profiles |
| `OPENFE_API_PROCESSORS` | `1` | Processes for partial charge generation |
| `OPENFE_API_CHECK_PARAMETERS` | `true` | Run the protein force field check during prep |
| `OPENFE_API_MAX_UPLOAD_BYTES` | `104857600` | Largest total size one campaign upload may carry |

Relative input paths in a request resolve against the service's working directory, so give
a container absolute paths to its mounted data.

!!! danger "The service has no authentication yet"
    There is no login, no token and no rate limit. Anyone who can reach the port can read
    every file in every campaign — your structures, your trajectories, your results — create
    new campaigns, submit jobs that spend your allocation, and name any input path the
    service process can read. Treat reaching the port as equivalent to a shell as the user
    running it.

    So: bind it to localhost and reach it through an SSH tunnel, or put it behind a reverse
    proxy that authenticates, on a network only your group can reach. The `--host 0.0.0.0`
    above is for a container whose port you publish deliberately, not for an open interface.
    Run it as a user who can reach nothing but its own data directory.

    Authentication is the next thing being added. Until it lands, network access control is
    the only thing protecting a campaign.

## Container images

One image carries both entry points.

```bash
docker build -f container/Dockerfile -t openfe-api:latest .

# the service
docker run --rm -p 8000:8000 -v /data:/data \
    -e OPENFE_API_ROOT=/data/campaigns openfe-api:latest

# the CLI
docker run --rm -v /data:/data openfe-api:latest \
    openfe-api status /data/campaigns/my_run
```

For HPC, `container/apptainer.def` builds the same environment directly, with no Docker
daemon involved — which is what most clusters need, since they do not run one:

```bash
apptainer build openfe-api.sif container/apptainer.def
apptainer run --nv --bind /work:/work openfe-api.sif \
    openfe-api submit /work/campaigns/my_run
```

`--nv` passes the GPU through, and is needed for any stage that simulates. Build it on a
compute node, not a login node: it installs several gigabytes and takes about 40 minutes.

!!! note "Submitting to the host scheduler from a container"
    To call the host's `sbatch` from inside a container you must bind the scheduler's
    client binaries, its configuration and its authentication socket into the container, and
    pass the relevant environment through. Which paths those are differs per cluster, so
    this is left to your own launch wrapper.

## Results are served per protocol

`POST /campaigns` takes any protocol's request; `protocol` selects the shape.

Each protocol answers a different question, so each has its own endpoint, and asking for the
wrong one returns `409` naming the right one. A network's per-ligand values are relative to the
network mean rather than absolute, so serving them from `/results` would invite them to be read
as absolute; plain MD has no free energy at all.

```bash
curl localhost:8000/campaigns/my_series/network
```

```json
{
  "edges": [
    {"edge": "lig_a_to_lig_b", "ddg": -1.20, "uncertainty": 0.18, "quality": "pass"}
  ],
  "ligands": [
    {"ligand": "lig_a", "dg_relative": -0.60, "uncertainty": 0.09, "edges": 2}
  ],
  "cycles": [
    {"ligands": ["lig_a", "lig_b", "lig_c"], "closure": 0.40, "per_edge": 0.23}
  ],
  "unreachable": [],
  "checks": [{"name": "cycle_closure", "status": "pass", "detail": "..."}]
}
```

A plain MD campaign answers with its files instead:

```bash
curl localhost:8000/campaigns/my_md/simulations
```

```json
[
  {
    "system": "holo",
    "repeat": 1,
    "finished": true,
    "artifacts": {
      "trajectory": "runs/holo/repeat1/shared_PlainMDSimulationUnit-.../simulation.xtc",
      "npt_structure": "runs/holo/repeat1/shared_PlainMDSimulationUnit-.../equil_npt.pdb"
    },
    "note": null
  }
]
```
