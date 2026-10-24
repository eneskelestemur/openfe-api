# How a campaign works

A campaign is one directory. It holds the request, everything built from it, and the state
of every run.

```text
my_campaign/
├── manifest.json        state of every run, plus provenance
├── inputs/              the campaign's copy of uploaded input files
├── prepared/            protein.pdb, the molecules, prep_report.json, charges/
├── plans/               transformation JSON, network.json, network.graphml for RBFE
├── runs/<run>/repeatN/  working directory, results.json, quickrun.log
├── relaxed/<name>/      the optional relaxation: its input, frame and report
├── results/             results.tsv for ABFE, edges.tsv and ligands.tsv for a network
├── analysis/            figures, their tables and the metrics
├── logs/                scheduler logs
└── submit_<class>.sh    the generated batch script, one per cost class
```

## Runs and tasks

A **run** is a complex for ABFE, a system for plain MD, and an edge for RBFE or SepTop. ABFE
and MD prepare each run into `prepared/<name>/`; a series shares one frame, so it prepares
into `prepared/` directly with its ligands under `prepared/ligands/`.

A **task** is one repeat of one run. An RBFE edge is two tasks, one per phase, under
`runs/<edge>/solvent/` and `runs/<edge>/complex/`. Everything else is one task per repeat,
because the protocol builds both phases itself.

## State

`manifest.json` is the source of truth. The CLI and the service both read and write it, so
work started through one is visible to the other. `openfe-api status` prints it.

`inputs/` only appears for a campaign created through the service's upload endpoint;
elsewhere the request points at files already on disk. Either way the service can list and
serve anything here, so a client with no filesystem access still gets its results. See
[getting files out](../running/service.md#getting-files-out).
