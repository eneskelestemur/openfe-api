# API reference

The library underneath the CLI and the service. Everything here is usable directly:

```python
from pathlib import Path
from openfe_api import Campaign, load_request

request = load_request(Path("request.yaml"))
campaign = Campaign.create(Path("./my_campaign"), request)
```

## Requests

::: openfe_api.schema.request

::: openfe_api.schema.common

::: openfe_api.schema.abfe

::: openfe_api.schema.series

::: openfe_api.schema.rbfe

::: openfe_api.schema.septop

::: openfe_api.schema.md

## Campaigns

::: openfe_api.campaign

## Preparation

::: openfe_api.prep.complex

::: openfe_api.prep.series

::: openfe_api.prep.system

::: openfe_api.prep.mcs

::: openfe_api.prep.gate

::: openfe_api.prep.pose

::: openfe_api.prep.align

::: openfe_api.prep.ligand

::: openfe_api.prep.protein

::: openfe_api.prep.structure

::: openfe_api.prep.checks

::: openfe_api.prep.report

::: openfe_api.prep.relax

::: openfe_api.prep.runner

## Planning

::: openfe_api.protocols.abfe

::: openfe_api.protocols.rbfe

::: openfe_api.protocols.septop

::: openfe_api.protocols.md

::: openfe_api.protocols.network

::: openfe_api.protocols.settings

::: openfe_api.protocols.charges

::: openfe_api.protocols.restraints

::: openfe_api.protocols.runner

## Execution

::: openfe_api.execution.base

::: openfe_api.execution.slurm

::: openfe_api.execution.local

::: openfe_api.schema.profiles

::: openfe_api.execution.mps

::: openfe_api.execution.runner

## Results

::: openfe_api.results.gather

::: openfe_api.results.network

::: openfe_api.results.septop

::: openfe_api.results.md

::: openfe_api.results.qc

::: openfe_api.results.runner

::: openfe_api.graph

::: openfe_api.files

## Supporting modules

::: openfe_api.config

::: openfe_api.log

::: openfe_api.provenance

## Errors

::: openfe_api.exceptions
