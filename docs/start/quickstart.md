# Your first campaign

The repository ships a campaign built from the test data. The "protein" is a twelve residue
fragment and the ligand is 1-phenylethanol, so the numbers mean nothing, but it exercises
every stage and needs no GPU.

```bash
openfe-api validate examples/abfe_smoke_test.yaml
openfe-api create   examples/abfe_smoke_test.yaml ./smoke
openfe-api prep     ./smoke
openfe-api plan     ./smoke
openfe-api status   ./smoke
```

Simulating needs a GPU and an [execution profile](../running/execution.md):

```bash
openfe-api submit  ./smoke --profiles profiles.yaml --dry-run   # read the script first
openfe-api submit  ./smoke --profiles profiles.yaml
openfe-api gather  ./smoke
openfe-api analyze ./smoke
```

## What each stage costs

| `validate`, `create` | `prep` | `plan` | `submit` | `gather`, `analyze` |
|---|---|---|---|---|
| instant | seconds | seconds to minutes | hours to days | seconds |

Only `submit` needs a GPU, and so does the optional `relax` stage, which runs between
`create` and `prep` when the request sets [`relax.enabled`](../protocols/md.md#relaxation).

Every stage records what it did in the campaign directory, so stages can run on different
machines: plan on a login node, submit to a cluster, gather anywhere.

## Next

Copy a template from `examples/`, then read the [input contract](../running/inputs.md) for
what each field means and [choosing a protocol](../protocols/index.md) for which to run.
