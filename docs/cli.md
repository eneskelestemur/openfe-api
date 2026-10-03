# CLI reference

```bash
openfe-api [--log-level LEVEL] [--log-file PATH] COMMAND ...
```

Errors go to standard error and results to standard output, so output can be piped.
`--log-file` additionally writes debug-level logs.

## The pipeline

| Command | Arguments | Purpose |
|---|---|---|
| `validate` | `REQUEST` | Check a request without touching any structure |
| `create` | `REQUEST CAMPAIGN_DIR` `[--force]` | Make a campaign directory |
| `relax` | `CAMPAIGN_DIR` `[--profiles F]` `[-p N]` `[--dry-run]` | Queue a short MD relaxation, if the request asks for one |
| `prep` | `CAMPAIGN_DIR` `[--only N]` `[--skip-parameter-check]` | Prepare inputs |
| `plan` | `CAMPAIGN_DIR` `[--only N]` `[-p N]` | Charges, transformations, cost, and a network's edges |
| `submit` | `CAMPAIGN_DIR` `[--profiles F]` `[--only N]` `[--dry-run]` `[--force]` | Run or queue repeats |
| `gather` | `CAMPAIGN_DIR` `[--only N]` `[--quiet]` | Results and quality checks |
| `status` | `CAMPAIGN_DIR` | State of every run |
| `version` | | Print the version |

`--only` is repeatable and restricts a command to the named runs. `prep` and `plan` reject it
for RBFE and SepTop, whose edges only exist once the network has been planned.

## Exit codes

`0` on success, `1` when a command fails or when any run within it failed, and `130` when a
local `submit` is interrupted. So `gather` exits non-zero when there is nothing usable to
report.

## A full session

```bash
openfe-api validate request.yaml
openfe-api create   request.yaml ./campaign
openfe-api prep     ./campaign
openfe-api plan     ./campaign -p 8
openfe-api submit   ./campaign --profiles profiles.yaml --dry-run   # read the script
openfe-api submit   ./campaign --profiles profiles.yaml
openfe-api status   ./campaign
openfe-api gather   ./campaign
```
