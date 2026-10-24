# Choosing a protocol

`protocol` in the request selects one of four. It also selects the shape of the rest of the
request: a field belonging to another protocol is rejected rather than ignored.

| | [ABFE](abfe.md) | [RBFE](rbfe.md) | [SepTop](septop.md) | [MD](md.md) |
|---|---|---|---|---|
| Answers | absolute ΔG | ΔΔG per pair | ΔΔG per pair | a trajectory |
| Needs an atom mapping | no | yes | no | no |
| Handles a scaffold hop | yes | no | yes | n/a |
| Net charge change | must be neutral | one, corrected | not supported | any |
| Cost per answer | ~484 ns | ~130 ns | ~510 ns | ~6 ns |
| Measured peak VRAM | ~8.4 GB | ~6.9 GB | ~13.6 GB | ~0.2 GB |
| Jobs per GPU, measured | 1 | 2 or 3 on 46 GB | 2 on 46 GB | 3+ |

[ABFE](abfe.md) and [MD](md.md) requests list independent runs. [RBFE](rbfe.md) and
[SepTop](septop.md) describe one series in one receptor, and their runs are the network edges
chosen during planning. Any request may also set [`relax`](md.md#relaxation) to run a short
MD of its input structure first.
