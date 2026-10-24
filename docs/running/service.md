# Service and container

The same library is an HTTP service, so a container can accept campaigns and submit them to
a cluster's scheduler.

```bash
uvicorn openfe_api.service:app --host 0.0.0.0 --port 8000
```

Interactive documentation is at `/docs`, generated from the models the CLI uses.

!!! danger "No authentication yet"
    There is no login, no token and no rate limit. Anyone who can reach the port can read
    every file in every campaign, create campaigns, and submit jobs that spend your
    allocation. Treat reaching the port as equivalent to a shell as the user running it.

    Bind it to localhost and reach it through an SSH tunnel, or put it behind a reverse proxy
    that authenticates. Run it as a user who can reach nothing but its own data directory.

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
| `POST` | `/campaigns/{name}/analysis` | Write figures and tables, and report the metrics |
| `GET` | `/campaigns/{name}/files` | List the files the campaign holds |
| `GET` | `/campaigns/{name}/files/{path}` | Download one of them |
| `GET` | `/campaigns/{name}/archive` | Download the campaign as one tar |

`POST /campaigns` takes any protocol's request, as JSON, with the same structure as the YAML.
Results are served per protocol, so asking for the wrong one returns `409` naming the right
one: a network's per-ligand values are relative to the network mean, and plain MD has no free
energy at all.

## Uploading inputs

`POST /campaigns` resolves input paths on the service's own filesystem. `POST
/campaigns/upload` takes the files with the request instead: a `campaign` form field holding
the JSON, and one `files` part per input.

```bash
curl --fail-with-body http://localhost:8000/campaigns/upload \
    --form-string "campaign=$(cat campaign.json)" \
    --form 'files=@bound.cif'
```

Every path in the request must be a bare filename matching an uploaded one, suffixed `.cif`,
`.mmcif`, `.pdb`, `.sdf`, `.mol` or `.mol2`. The campaign keeps its own copy under `inputs/`.
An existing campaign is never overwritten, and a failed upload leaves no directory behind.
Chemistry is validated during `prep`, not here.

!!! warning "Set a body limit on your proxy too"
    `OPENFE_API_MAX_UPLOAD_BYTES` is enforced while the files are written, but the multipart
    parser has already spooled the request by then.

## Getting files out

A client with no filesystem access reads everything back over HTTP:

```bash
curl localhost:8000/campaigns/my_run/files?group=results
curl -O localhost:8000/campaigns/my_run/files/results/results.tsv
curl -O localhost:8000/campaigns/my_run/archive
```

The groups are `inputs`, `prepared`, `plans`, `relaxed`, `runs`, `results` and `logs`. Repeat
`group=` for several, or leave it off for everything. Each listed `path` goes straight into
the download endpoint, and a plain MD campaign's `/simulations` response reports its
artifacts as these same paths.

| | Archive | Single files |
|---|---|---|
| Carries | every group **except `runs`**; add `?group=runs` for it | whatever you list |
| Compression | none, so `Content-Length` is exact; `?compress=true` to gzip | none |
| Resumable | no | yes, ranged requests |
| Use for | a finished campaign's tables and reports | anything large |

For anything large, pull files individually and let a downloader retry and parallelize:

```bash
BASE=localhost:8000/campaigns/my_run
curl -s "$BASE/files?group=runs" \
    | python -c 'import json,sys;[print(f["path"]) for f in json.load(sys.stdin)]' \
    | sed "s|^|$BASE/files/|" > urls.txt
aria2c -x8 -i urls.txt          # or: wget -c -i urls.txt
```

## Long operations

`prep`, `plan` and `submit` return `202 Accepted` and run in the background. Progress comes
from `GET /campaigns/{name}`, which reports every run's state and a `last_operation`
record:

```json
{
  "operation": "plan",
  "status": "failed",
  "started_at": "2026-09-16T10:00:00Z",
  "finished_at": "2026-09-16T10:00:04Z",
  "message": "planned 0, failed 1, 0 ns total"
}
```

So a failure inside a background task is still visible. One campaign runs one operation at a
time; a duplicate request while it is busy is ignored and logged.

The analysis endpoint answers in the request instead. With `trajectories` set that is minutes
to hours on a production campaign: raise `workers` and set a client timeout to match, or use
[`analyze --submit`](../results/analysis.md#running-it-on-a-cluster) to hand it to Slurm.

## Configuration

Every setting is an environment variable prefixed with `OPENFE_API_`. See `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `OPENFE_API_ROOT` | `campaigns` | Directory holding one subdirectory per campaign |
| `OPENFE_API_PROFILES_FILE` | unset | YAML file of execution profiles |
| `OPENFE_API_PROCESSORS` | `1` | Processes for partial charge generation |
| `OPENFE_API_CHECK_PARAMETERS` | `true` | Run the protein force field check during prep |
| `OPENFE_API_MAX_UPLOAD_BYTES` | `104857600` | Largest total size one campaign upload may carry |

Relative input paths resolve against the service's working directory, so give a container
absolute paths to its mounted data.

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

`container/apptainer.def` builds the same environment with no Docker daemon:

```bash
apptainer build openfe-api.sif container/apptainer.def
apptainer run --nv --bind /work:/work openfe-api.sif \
    openfe-api submit /work/campaigns/my_run
```

`--nv` passes the GPU through and is needed for any stage that simulates. Build on a compute
node: it installs several gigabytes and takes about 40 minutes.

!!! note "Submitting to the host scheduler from a container"
    Calling the host's `sbatch` from inside a container means binding the scheduler's client
    binaries, its configuration and its authentication socket into the container, and passing
    the relevant environment through. Which paths those are differs per cluster, so this is
    left to your own launch wrapper.
