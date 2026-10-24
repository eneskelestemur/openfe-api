# CLI reference

```bash
openfe-api [--log-level LEVEL] [--log-file PATH] COMMAND ...
```

Errors go to standard error and results to standard output, so output can be piped.
`--log-file` additionally writes debug-level logs. What each stage does is on
[the landing page](../index.md); a worked sequence is in
[your first campaign](../start/quickstart.md).

| Command | Arguments |
|---|---|
| `validate` | `REQUEST` |
| `create` | `REQUEST CAMPAIGN_DIR` `[--force]` |
| `relax` | `CAMPAIGN_DIR` `[--profiles F]` `[-p N]` `[--dry-run]` |
| `prep` | `CAMPAIGN_DIR` `[--only N]` `[--skip-parameter-check]` |
| `plan` | `CAMPAIGN_DIR` `[--only N]` `[-p N]` |
| `submit` | `CAMPAIGN_DIR` `[--profiles F]` `[--only N]` `[--dry-run]` `[--force]` |
| `gather` | `CAMPAIGN_DIR` `[--only N]` `[--quiet]` |
| `analyze` | `CAMPAIGN_DIR` `[--measured F]` `[--dark]` `[--trajectories]` `[--workers N]` `[--submit]` `[--profile P]` `[--profiles F]` `[--dry-run]` |
| `status` | `CAMPAIGN_DIR` |
| `version` | |

Beyond the five pipeline stages: `validate` checks a request without touching a structure,
`create` makes the campaign directory, `relax` queues the optional
[relaxation](../protocols/md.md#relaxation), `status` prints the state of every run, and
`version` prints the version.

`--only` is repeatable and restricts a command to the named runs. `prep` and `plan` reject it
for RBFE and SepTop, whose edges only exist once the network has been planned.

## Exit codes

`0` on success, `1` when a command fails or when any run within it failed, and `130` when a
local `submit` is interrupted. So `gather` exits non-zero when there is nothing usable to
report.
