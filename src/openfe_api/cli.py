"""Command line interface for openfe-api.

This module is a thin layer over the library: it parses arguments, renders output, and
maps errors onto exit codes. All behavior lives in the library modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from openfe_api import __version__
from openfe_api.campaign import Campaign, RunState
from openfe_api.exceptions import OpenFEAPIError
from openfe_api.execution.runner import load_profile, relax_campaign, submit_campaign
from openfe_api.log import configure_logging
from openfe_api.prep.report import SeriesPrepReport
from openfe_api.prep.runner import prepare_campaign
from openfe_api.protocols.runner import PlanOutcome, plan_campaign
from openfe_api.results.runner import GatherOutcome, gather_campaign
from openfe_api.schema.abfe import AbfeRequest
from openfe_api.schema.md import MdRequest
from openfe_api.schema.rbfe import RbfeRequest
from openfe_api.schema.request import CampaignRequest, load_request
from openfe_api.schema.septop import SepTopRequest

__all__ = ["app"]

app = typer.Typer(
    name="openfe-api",
    help="Run OpenFE free energy experiments on HPC and server VMs.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
error_console = Console(stderr=True)

_STATE_STYLES: dict[RunState, str] = {
    RunState.PENDING: "dim",
    RunState.PREPARED: "cyan",
    RunState.PLANNED: "blue",
    RunState.SUBMITTED: "yellow",
    RunState.RUNNING: "bold yellow",
    RunState.DONE: "green",
    RunState.FAILED: "bold red",
}

RequestArgument = Annotated[
    Path,
    typer.Argument(exists=False, dir_okay=False, help="Path to the campaign request YAML file."),
]
CampaignArgument = Annotated[
    Path,
    typer.Argument(file_okay=False, help="Path to the campaign directory."),
]


def _fail(error: OpenFEAPIError) -> None:
    """Render an error and exit with a non-zero status."""
    error_console.print(Panel(str(error), title="Invalid input", border_style="red"))
    raise typer.Exit(code=1)


def _load_request(path: Path) -> CampaignRequest:
    """Load a request file, exiting with a rendered error if it is invalid."""
    try:
        return load_request(path)
    except OpenFEAPIError as error:
        _fail(error)
        raise  # unreachable; keeps the return type honest


@app.callback()
def main(
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Console log level.", metavar="LEVEL"),
    ] = "INFO",
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Also write debug-level logs to this file."),
    ] = None,
) -> None:
    """Configure logging for every command.

    Args:
        log_level: Console log level name, such as ``DEBUG`` or ``INFO``.
        log_file: Optional file receiving debug-level logs.

    Raises:
        typer.BadParameter: If the log level name is not recognized.
    """
    level = logging.getLevelNamesMapping().get(log_level.upper())
    if level is None:
        raise typer.BadParameter(f"unknown log level '{log_level}'", param_hint="--log-level")
    configure_logging(level=level, log_file=log_file, console=error_console)


@app.command()
def version() -> None:
    """Print the openfe-api version."""
    console.print(__version__)


def _render_abfe_request(request: AbfeRequest) -> None:
    """Print a table of an ABFE request's complexes and its neutralization warning."""
    table = Table(title=f"{request.name} ({request.protocol})", header_style="bold")
    table.add_column("Complex")
    table.add_column("Ligand")
    table.add_column("Cofactors")
    table.add_column("Extra copies")
    for complex_spec in request.complexes:
        table.add_row(
            complex_spec.name,
            complex_spec.ligand_name(),
            ", ".join(cofactor.name for cofactor in complex_spec.cofactors) or "-",
            complex_spec.extra_ligand_copies or "-",
        )
    console.print(table)
    console.print(
        f"Profile [bold]{request.execution.profile or 'local'}[/bold], "
        f"{request.execution.repeats} repeat(s) per complex, "
        f"settings preset [bold]{request.settings.preset}[/bold]."
    )
    if request.neutralize_ligands:
        console.print(
            Panel(
                "Ligand neutralization is ENABLED. Neutralizing changes the molecule "
                "chemically and the computed affinity will refer to the neutral form, not "
                "the species you specified.",
                title="Warning",
                border_style="yellow",
            )
        )


def _render_series(request: RbfeRequest | SepTopRequest) -> None:
    """Print a table of a series request's ligands and how each pose will be obtained."""
    reference = request.reference_ligand()
    poses = request.requested_poses()
    table = Table(title=f"{request.name} ({request.protocol})", header_style="bold")
    table.add_column("Ligand")
    table.add_column("Role")
    table.add_column("Pose")
    table.add_column("Source")
    for ligand in request.ligands:
        source = ligand.structure or ligand.path
        table.add_row(
            ligand.name,
            "reference" if ligand.name == reference.name else "-",
            poses[ligand.name],
            source.name if source is not None else "SMILES only",
        )
    console.print(table)
    console.print(
        f"Profile [bold]{request.execution.profile or 'local'}[/bold], "
        f"{request.execution.repeats} repeat(s) per edge, "
        f"settings preset [bold]{request.settings.preset}[/bold]."
    )


def _render_rbfe_request(request: RbfeRequest) -> None:
    """Print an RBFE request's series, its network and its charge policy."""
    _render_series(request)
    console.print(
        f"Network [bold]{request.network.method}[/bold] with mapper "
        f"[bold]{request.network.mapper}[/bold], common core at least "
        f"[bold]{request.similarity.min_core_fraction:.0%}[/bold] of the smaller ligand."
    )
    if not request.charges.correct_single:
        console.print(
            Panel(
                "Charge correction is DISABLED. Edges that change the net charge by one "
                "will run uncorrected, which biases their result.",
                title="Warning",
                border_style="yellow",
            )
        )
    if request.charges.allow_multi:
        console.print(
            Panel(
                "Edges changing the net charge by more than one are ALLOWED. OpenFE has no "
                "correction for these, so their results are unreliable.",
                title="Warning",
                border_style="yellow",
            )
        )


def _render_septop_request(request: SepTopRequest) -> None:
    """Print a SepTop request's series, its network and its screening thresholds."""
    _render_series(request)
    hub = f" on hub [bold]{request.hub()}[/bold]" if request.network.method != "explicit" else ""
    console.print(f"Network [bold]{request.network.method}[/bold]{hub}.")
    core = (
        f"at least [bold]{request.screening.min_core_fraction:.0%}[/bold] common core"
        if request.screening.min_core_fraction is not None
        else "no common core required"
    )
    console.print(
        f"Screening [bold]{request.screening.policy}[/bold]: {core}, size ratio at most "
        f"[bold]{request.screening.max_size_ratio:.2f}[/bold], at most "
        f"[bold]{request.screening.max_clashes}[/bold] clashing atom(s). A net charge change "
        "is always excluded, because the protocol does not support one."
    )


def _render_md_request(request: MdRequest) -> None:
    """Print a table of a plain MD request's systems and what each one holds."""
    table = Table(title=f"{request.name} ({request.protocol})", header_style="bold")
    table.add_column("System")
    table.add_column("Protein")
    table.add_column("Molecules")
    table.add_column("Solvent")
    for system in request.systems:
        table.add_row(
            system.name,
            "yes" if system.protein is not None else "-",
            ", ".join(molecule.name for molecule in system.molecules()) or "-",
            "yes" if system.solvent else "[yellow]vacuum[/yellow]",
        )
    console.print(table)
    console.print(
        f"Profile [bold]{request.execution.profile or 'local'}[/bold], "
        f"{request.execution.repeats} independent run(s) per system, "
        f"settings preset [bold]{request.settings.preset}[/bold]."
    )
    if any(not system.solvent for system in request.systems):
        console.print(
            Panel(
                "A system runs in vacuum, so planning sets its nonbonded method to "
                "'nocutoff'. PME needs a periodic box and the protocol would refuse the "
                "system with it.",
                title="Note",
                border_style="yellow",
            )
        )


def _render_request(request: CampaignRequest) -> None:
    """Print the request belonging to whichever protocol it names."""
    match request:
        case AbfeRequest():
            _render_abfe_request(request)
        case RbfeRequest():
            _render_rbfe_request(request)
        case SepTopRequest():
            _render_septop_request(request)
        case MdRequest():
            _render_md_request(request)
        case _:
            raise typer.BadParameter(f"unknown protocol '{request.protocol}'")


@app.command()
def validate(request_file: RequestArgument) -> None:
    """Validate a campaign request without touching any input structure.

    Args:
        request_file: Path to the campaign request YAML file.
    """
    _render_request(_load_request(request_file))
    console.print("[green]Request is valid.[/green]")


@app.command()
def create(
    request_file: RequestArgument,
    campaign_dir: CampaignArgument,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing campaign in the directory."),
    ] = False,
) -> None:
    """Create a campaign directory from a validated request.

    Args:
        request_file: Path to the campaign request YAML file.
        campaign_dir: Directory to create the campaign in.
        force: Whether to overwrite an existing campaign.
    """
    request = _load_request(request_file)
    try:
        campaign = Campaign.create(campaign_dir, request, exist_ok=force)
    except OpenFEAPIError as error:
        _fail(error)
        return

    console.print(
        f"Created campaign [bold]{campaign.manifest.name}[/bold] with "
        f"{len(campaign.manifest.runs)} run(s) at {campaign.directory}"
    )


def _render_series_prep(report: SeriesPrepReport) -> None:
    """Print how each ligand of a series was placed, and which were excluded."""
    table = Table(title=f"{report.name}: series preparation", header_style="bold")
    table.add_column("Ligand")
    table.add_column("Pose")
    table.add_column("Core")
    table.add_column("Fit")
    table.add_column("Clashes")
    for placement in report.ligands:
        role = " (reference)" if placement.name == report.reference else ""
        core = (
            f"{placement.core_size} atoms, {placement.core_fraction:.0%}"
            if placement.core_size is not None and placement.core_fraction is not None
            else "none shared"
        )
        if placement.core_rmsd is not None:
            fit = f"core RMSD {placement.core_rmsd:.3f} A"
        elif placement.shape_score is not None:
            fit = f"shape score {placement.shape_score:.1f}"
        else:
            fit = "-"
        table.add_row(f"{placement.name}{role}", placement.pose, core, fit, str(placement.clashes))
    console.print(table)

    if report.cofactors:
        console.print(f"Cofactors: {', '.join(cofactor.name for cofactor in report.cofactors)}.")
    if report.dropped:
        excluded = Table(title="Excluded ligands", header_style="bold")
        excluded.add_column("Ligand")
        excluded.add_column("Reason")
        for name, reason in sorted(report.dropped.items()):
            excluded.add_row(name, reason)
        console.print(excluded)


@app.command()
def relax(
    campaign_dir: CampaignArgument,
    profiles_file: Annotated[
        Path | None,
        typer.Option("--profiles", help="YAML file of execution profiles."),
    ] = None,
    processors: Annotated[
        int,
        typer.Option("--processors", "-p", min=1, help="Processes for partial charges."),
    ] = 1,
    skip_parameter_check: Annotated[
        bool,
        typer.Option("--skip-parameter-check", help="Skip the force field check on the protein."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Write the batch script without submitting it."),
    ] = False,
) -> None:
    """Relax a campaign's input structures with a short MD run, before preparing them.

    Only campaigns whose request sets ``relax.enabled`` use this stage. It prepares the
    system that carries the frame, plans a short MD run of it, and queues that run; once the
    job finishes, ``prep`` reads the relaxed structure instead of the original.

    Args:
        campaign_dir: Path to the campaign directory.
        profiles_file: YAML file holding execution profiles.
        processors: Number of processes to use for partial charge generation.
        skip_parameter_check: Whether to skip the protein force field check.
        dry_run: For Slurm, write the batch script without submitting it.
    """
    try:
        campaign = Campaign.open(campaign_dir)
        profile = load_profile(campaign.manifest.request.execution.profile, profiles_file)
        outcome = relax_campaign(
            campaign,
            profile,
            check_parameters=not skip_parameter_check,
            processors=processors,
            dry_run=dry_run,
        )
    except OpenFEAPIError as error:
        _fail(error)
        return

    table = Table(title=f"{campaign.manifest.name}: relaxation", header_style="bold")
    table.add_column("System")
    table.add_column("State")
    for name in outcome.targets:
        if name in outcome.skipped:
            state = "[green]already relaxed[/green]"
        elif any(task.run_name == name for task in outcome.tasks):
            state = "[blue]queued[/blue]" if outcome.job_id else "[blue]run[/blue]"
        else:
            state = "[yellow]nothing to do[/yellow]"
        table.add_row(name, state)
    console.print(table)

    if outcome.job_id is not None:
        console.print(f"Queued as job [bold]{outcome.job_id}[/bold].")
    if outcome.script is not None and outcome.job_id is None:
        console.print(f"Wrote {outcome.script} without submitting it.")
    if outcome.tasks:
        console.print(
            "Run [bold]openfe-api prep[/bold] once the relaxation has finished; it reads the "
            "relaxed structure in place of the original."
        )


@app.command()
def prep(
    campaign_dir: CampaignArgument,
    only: Annotated[
        list[str] | None,
        typer.Option(
            "--only",
            help="Prepare only these runs. Repeatable. Not for a network protocol.",
        ),
    ] = None,
    skip_parameter_check: Annotated[
        bool,
        typer.Option(
            "--skip-parameter-check",
            help="Skip the force field check on each prepared protein.",
        ),
    ] = False,
) -> None:
    """Prepare the inputs of a campaign for simulation.

    Args:
        campaign_dir: Path to the campaign directory.
        only: Run names to prepare; everything if omitted. ABFE and plain MD only.
        skip_parameter_check: Whether to skip the protein force field check.
    """
    try:
        campaign = Campaign.open(campaign_dir)
        outcome = prepare_campaign(
            campaign, only=only or None, check_parameters=not skip_parameter_check
        )
    except OpenFEAPIError as error:
        _fail(error)
        return

    if outcome.series is not None:
        _render_series_prep(outcome.series)
    elif outcome.systems:
        table = Table(title=f"{campaign.manifest.name}: preparation", header_style="bold")
        table.add_column("System")
        table.add_column("Result")
        table.add_column("Detail")
        for name, system in outcome.systems.items():
            protein = (
                f"protein {system.protein.n_atoms} atoms"
                if system.protein is not None
                else "no protein"
            )
            table.add_row(
                name,
                "[green]prepared[/green]",
                f"{protein}, {len(system.molecules)} molecule(s)",
            )
        for name, message in outcome.failures.items():
            table.add_row(name, "[bold red]failed[/bold red]", message.splitlines()[0])
        console.print(table)
    else:
        table = Table(title=f"{campaign.manifest.name}: preparation", header_style="bold")
        table.add_column("Complex")
        table.add_column("Result")
        table.add_column("Detail")
        for name, report in outcome.reports.items():
            detail = (
                f"protein {report.protein.n_atoms} atoms, "
                f"ligand {report.ligand.n_atoms} atoms, "
                f"{len(report.cofactors)} cofactor(s)"
            )
            table.add_row(name, "[green]prepared[/green]", detail)
        for name, message in outcome.failures.items():
            table.add_row(name, "[bold red]failed[/bold red]", message.splitlines()[0])
        console.print(table)

    for warning in outcome.warnings:
        console.print(f"[yellow]warning[/yellow] {warning}")

    if outcome.failures:
        raise typer.Exit(code=1)


def _render_exclusions(outcome: PlanOutcome) -> None:
    """Print what planning excluded from a network, and anything it left unreachable."""
    network = outcome.network
    if network is None:
        return

    if network.dropped:
        excluded = Table(title="Excluded", header_style="bold")
        excluded.add_column("Name")
        excluded.add_column("Reason")
        for excluded_name, reason in sorted(network.dropped.items()):
            excluded.add_row(excluded_name, reason)
        console.print(excluded)

    if network.unreachable:
        console.print(
            Panel(
                "These ligands have no path to the rest of the network, so they will get no "
                f"comparable free energy: {', '.join(network.unreachable)}.",
                title="Warning",
                border_style="yellow",
            )
        )


def _render_septop_plan(name: str, outcome: PlanOutcome) -> None:
    """Print the planned SepTop edges and their cost."""
    table = Table(title=f"{name}: planned network", header_style="bold")
    table.add_column("Edge")
    table.add_column("Windows")
    table.add_column("Cost")
    for edge in outcome.septop_edges:
        table.add_row(
            edge.name,
            f"{edge.complex_replicas} complex / {edge.solvent_replicas} solvent",
            f"{edge.cost.total_ns:,.0f} ns",
        )
    console.print(table)
    _render_exclusions(outcome)


def _render_network_plan(name: str, outcome: PlanOutcome) -> None:
    """Print the planned RBFE edges, their charge decisions, and what was excluded."""
    table = Table(title=f"{name}: planned network", header_style="bold")
    table.add_column("Edge")
    table.add_column("Charge")
    table.add_column("Score")
    table.add_column("Windows")
    table.add_column("Cost")
    for edge in outcome.edges:
        difference = edge.decision.charge_difference
        if difference == 0:
            charge = "-"
        elif edge.decision.corrected:
            charge = f"[yellow]{difference:+d} corrected[/yellow]"
        else:
            charge = f"[bold red]{difference:+d} uncorrected[/bold red]"
        table.add_row(
            edge.name,
            charge,
            f"{edge.decision.score:.2f}",
            str(edge.n_replicas),
            f"{edge.cost.total_ns:,.0f} ns",
        )
    console.print(table)
    _render_exclusions(outcome)


@app.command()
def plan(
    campaign_dir: CampaignArgument,
    only: Annotated[
        list[str] | None,
        typer.Option(
            "--only",
            help="Plan only these runs. Repeatable. Not for a network protocol.",
        ),
    ] = None,
    processors: Annotated[
        int,
        typer.Option("--processors", "-p", min=1, help="Processes for partial charges."),
    ] = 1,
) -> None:
    """Plan the transformations a campaign will run.

    Args:
        campaign_dir: Path to the campaign directory.
        only: Run names to plan; everything if omitted. ABFE and plain MD only.
        processors: Number of processes to use for partial charge generation.
    """
    try:
        campaign = Campaign.open(campaign_dir)
        outcome = plan_campaign(campaign, only=only or None, processors=processors)
    except OpenFEAPIError as error:
        _fail(error)
        return

    if outcome.md_runs:
        table = Table(title=f"{campaign.manifest.name}: planned simulations", header_style="bold")
        table.add_column("System")
        table.add_column("Components", justify="right")
        table.add_column("Solvent")
        table.add_column("Cost")
        for run in outcome.md_runs:
            table.add_row(
                run.name,
                str(run.n_components),
                "yes" if run.solvated else "vacuum",
                run.cost.describe(),
            )
        for name, message in outcome.failures.items():
            table.add_row(name, "-", "-", f"[bold red]{message.splitlines()[0]}[/bold red]")
        console.print(table)
    elif outcome.septop_edges:
        _render_septop_plan(campaign.manifest.name, outcome)
    elif outcome.edges or outcome.network is not None:
        _render_network_plan(campaign.manifest.name, outcome)
    else:
        table = Table(title=f"{campaign.manifest.name}: planning", header_style="bold")
        table.add_column("Complex")
        table.add_column("Result")
        table.add_column("Detail")
        for name, planned in outcome.planned.items():
            table.add_row(name, "[blue]planned[/blue]", planned.cost.describe())
        for name, message in outcome.failures.items():
            table.add_row(name, "[bold red]failed[/bold red]", message.splitlines()[0])
        console.print(table)

    if outcome.planned or outcome.edges or outcome.septop_edges or outcome.md_runs:
        console.print(
            f"Total simulation time across the campaign: [bold]{outcome.total_ns:,.0f} ns[/bold]."
        )

    if outcome.failures:
        raise typer.Exit(code=1)


@app.command()
def submit(
    campaign_dir: CampaignArgument,
    profiles_file: Annotated[
        Path | None,
        typer.Option("--profiles", help="YAML file of execution profiles."),
    ] = None,
    only: Annotated[
        list[str] | None,
        typer.Option("--only", help="Submit only these runs. Repeatable."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rerun repeats that already have results."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Write the batch script without submitting it."),
    ] = False,
) -> None:
    """Run or queue every planned repeat of a campaign.

    Args:
        campaign_dir: Path to the campaign directory.
        profiles_file: YAML file holding execution profiles.
        only: Run names to submit; every planned run if omitted.
        force: Whether to rerun repeats that already have results.
        dry_run: Whether to only write the batch script, for the Slurm backend.
    """
    try:
        campaign = Campaign.open(campaign_dir)
        profile = load_profile(campaign.manifest.request.execution.profile, profiles_file)
        outcome = submit_campaign(
            campaign, profile, only=only or None, force=force, dry_run=dry_run
        )
    except OpenFEAPIError as error:
        _fail(error)
        return

    if outcome.backend == "slurm":
        if outcome.job_id is None:
            console.print(f"Dry run: wrote [bold]{outcome.script}[/bold]")
        else:
            console.print(
                f"Submitted [bold]{len(outcome.tasks)}[/bold] repeat(s) as job "
                f"[bold]{outcome.job_id}[/bold]"
            )
        return

    console.print(
        f"Finished [green]{len(outcome.completed)}[/green] repeat(s), "
        f"skipped {len(outcome.skipped)}, "
        f"[red]{len(outcome.failed)}[/red] failed, "
        f"[yellow]{len(outcome.interrupted)}[/yellow] interrupted"
    )
    for label, message in outcome.failed.items():
        console.print(f"[red]failed[/red] {label}: {message}")
    if outcome.interrupted:
        console.print("[yellow]Interrupted.[/yellow] Run the same submit command again to resume.")
        raise typer.Exit(code=130)
    if outcome.failed:
        raise typer.Exit(code=1)


def _render_md_results(name: str, outcome: GatherOutcome) -> None:
    """Print what each MD repeat produced, which is files rather than free energies."""
    table = Table(title=f"{name}: simulations", header_style="bold")
    table.add_column("System")
    table.add_column("Repeat", justify="right")
    table.add_column("Status")
    table.add_column("Trajectory")
    table.add_column("Equilibrated")
    for run in outcome.md:
        if not run.repeats:
            table.add_row(run.name, "-", "[yellow]no results yet[/yellow]", "-", "-")
            continue
        for entry in run.repeats:
            trajectory = entry.artifacts.get("trajectory")
            equilibrated = entry.artifacts.get("npt_structure")
            table.add_row(
                run.name,
                str(entry.repeat),
                "[green]finished[/green]" if entry.ok else "[bold red]failed[/bold red]",
                trajectory.name if trajectory is not None else "-",
                equilibrated.name if equilibrated is not None else "-",
            )
    console.print(table)
    console.print(
        "Plain MD reports no free energy. Trajectory analysis is a separate stage; the paths "
        "above are where the files live."
    )
    for run in outcome.md:
        for entry in run.repeats:
            if entry.failure is not None:
                console.print(
                    f"[red]failed[/red] {run.name} repeat {entry.repeat}: {entry.failure}"
                )


def _render_network_results(name: str, outcome: GatherOutcome, quiet: bool) -> None:
    """Print the per-edge and per-ligand results of an RBFE campaign."""
    results = outcome.network
    if results is None:
        return

    edges = Table(title=f"{name}: relative binding free energies", header_style="bold")
    edges.add_column("Edge")
    edges.add_column("ddG (kcal/mol)", justify="right")
    edges.add_column("uncertainty", justify="right")
    edges.add_column("from", justify="center")
    edges.add_column("quality")
    for edge in results.edges:
        report = outcome.reports.get(edge.name)
        verdict = report.verdict if report is not None else "unknown"
        style = {"pass": "green", "fail": "bold red", "unknown": "yellow"}[verdict]
        edges.add_row(
            edge.name,
            "-" if edge.ddg is None else f"{edge.ddg:+.2f}",
            "-" if edge.uncertainty is None else f"{edge.uncertainty:.2f}",
            edge.uncertainty_kind or "-",
            f"[{style}]{verdict}[/{style}]",
        )
    console.print(edges)

    if results.ligands:
        ligands = Table(
            title="Per-ligand free energies, relative to the network mean",
            header_style="bold",
        )
        ligands.add_column("Ligand")
        ligands.add_column("dG (kcal/mol)", justify="right")
        ligands.add_column("uncertainty", justify="right")
        ligands.add_column("edges", justify="right")
        for ligand in results.ligands:
            ligands.add_row(
                ligand.name,
                f"{ligand.dg:+.2f}",
                f"{ligand.uncertainty:.2f}",
                str(ligand.degree),
            )
        console.print(ligands)
        console.print(
            "These are fitted over the network, so differences between ligands are "
            "meaningful while a single value on its own is not."
        )

    if results.cycles:
        cycles = Table(title="Cycle closure", header_style="bold")
        cycles.add_column("Cycle")
        cycles.add_column("closes to (kcal/mol)", justify="right")
        cycles.add_column("per edge", justify="right")
        for cycle in results.cycles:
            cycles.add_row(
                " to ".join(cycle.ligands),
                f"{cycle.closure:+.2f}",
                f"{cycle.per_edge:+.2f}",
            )
        console.print(cycles)

    if not quiet:
        for check_name, report in outcome.reports.items():
            for check in report.checks:
                if check.status == "pass":
                    continue
                style = "red" if check.failed else "yellow"
                console.print(
                    f"[{style}]{check.status}[/{style}] {check_name} {check.name}: {check.detail}"
                )


@app.command()
def gather(
    campaign_dir: CampaignArgument,
    only: Annotated[
        list[str] | None,
        typer.Option("--only", help="Gather only these runs. Repeatable."),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Print the table only, without the quality checks."),
    ] = False,
) -> None:
    """Collect what the campaign produced: free energies, or files for plain MD.

    Args:
        campaign_dir: Path to the campaign directory.
        only: Run names to gather; everything if omitted.
        quiet: Whether to suppress the per-ligand quality checks.
    """
    try:
        campaign = Campaign.open(campaign_dir)
        outcome = gather_campaign(campaign, only=only or None)
    except OpenFEAPIError as error:
        _fail(error)
        return

    if outcome.md:
        _render_md_results(campaign.manifest.name, outcome)
        if not any(run.finished for run in outcome.md):
            raise typer.Exit(code=1)
        return

    if outcome.network is not None:
        _render_network_results(campaign.manifest.name, outcome, quiet=quiet)
        if not outcome.network.usable_edges:
            raise typer.Exit(code=1)
        return

    if not outcome.results:
        console.print("[yellow]No usable results yet.[/yellow]")
        for repeat in outcome.failed:
            console.print(f"[red]unusable[/red] {repeat.path}: {repeat.failure}")
        raise typer.Exit(code=1)

    table = Table(title=f"{campaign.manifest.name}: binding free energies", header_style="bold")
    table.add_column("Ligand")
    table.add_column("dG (kcal/mol)", justify="right")
    table.add_column("uncertainty", justify="right")
    table.add_column("from", justify="center")
    table.add_column("repeats", justify="right")
    table.add_column("quality")
    for result in outcome.results:
        report = outcome.reports[result.name]
        style = {"pass": "green", "fail": "bold red", "unknown": "yellow"}[report.verdict]
        table.add_row(
            result.name,
            f"{result.dg:.2f}",
            f"{result.uncertainty:.2f}",
            result.uncertainty_kind,
            str(len(result.repeats)),
            f"[{style}]{report.verdict}[/{style}]",
        )
    console.print(table)

    if not quiet:
        for name, report in outcome.reports.items():
            for check in report.checks:
                if check.status == "pass":
                    continue
                style = "red" if check.failed else "yellow"
                console.print(
                    f"[{style}]{check.status}[/{style}] {name} {check.name}: {check.detail}"
                )

    for repeat in outcome.failed:
        console.print(f"[red]unusable[/red] {repeat.path}: {repeat.failure}")
    if outcome.missing:
        console.print(f"[dim]no results yet for: {', '.join(outcome.missing)}[/dim]")
    if outcome.tsv is not None:
        console.print(f"Wrote {outcome.tsv}")


@app.command()
def status(campaign_dir: CampaignArgument) -> None:
    """Show the state of every run in a campaign.

    Args:
        campaign_dir: Path to the campaign directory.
    """
    try:
        campaign = Campaign.open(campaign_dir)
    except OpenFEAPIError as error:
        _fail(error)
        return

    table = Table(title=campaign.manifest.name, header_style="bold")
    table.add_column("Run")
    table.add_column("State")
    table.add_column("Repeats", justify="right")
    table.add_column("Job")
    table.add_column("Detail")
    for record in campaign.manifest.runs.values():
        table.add_row(
            record.name,
            f"[{_STATE_STYLES[record.state]}]{record.state.value}[/]",
            str(record.repeats),
            record.job_id or "-",
            record.message or "",
        )

    console.print(table)
    summary = ", ".join(
        f"{count} {state.value}" for state, count in campaign.counts().items() if count
    )
    console.print(summary)
