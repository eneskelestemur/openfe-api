"""The optional structure relaxation stage, which runs before preparation.

A short plain MD run of whichever system carries the campaign's frame, handing on the NPT frame.

The relaxed structure re-enters through preparation rather than being spliced into the prepared
files, so a residue mapped wrongly here fails against its declared SMILES instead of quietly
relaxing into the wrong chemistry.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from openfe_api.campaign import RELAXED_DIR, Campaign
from openfe_api.exceptions import InputValidationError, OpenFEAPIError
from openfe_api.execution.base import Task
from openfe_api.log import get_logger
from openfe_api.prep.structure import AMINO_ACIDS, IONS, TERMINAL_CAPS, WATERS, Structure
from openfe_api.prep.system import prepare_system
from openfe_api.protocols.charges import ChargeCache
from openfe_api.protocols.md import plan_md
from openfe_api.results.md import read_md_repeat
from openfe_api.schema.abfe import AbfeRequest
from openfe_api.schema.common import MoleculeSpec, RelaxSpec, Selector, SettingsSpec
from openfe_api.schema.md import MdLigandSpec, MdRequest, MdSystemSpec
from openfe_api.schema.rbfe import RbfeRequest
from openfe_api.schema.request import CampaignRequest
from openfe_api.schema.septop import SepTopRequest

__all__ = [
    "RELAX_CLASS",
    "RelaxOutcome",
    "RelaxReport",
    "check_relaxation",
    "collect_relaxed",
    "prepare_relaxation",
    "relax_targets",
    "relax_tasks",
    "relaxed_request",
]

logger = get_logger(__name__)

RELAX_CLASS = "relax"
"""Cost class of a relaxation, so it queues with its own wall-time limit."""

MINIMUM_PRODUCTION = "0.01 nanosecond"
"""Production length a relaxation runs.

The settings model requires a positive production length, and a relaxation hands on the
structure after NPT equilibration rather than anything sampled, so this is as short as the
model allows rather than a length chosen for the physics.
"""

NON_MOLECULE_RESIDUES = AMINO_ACIDS | WATERS | IONS | TERMINAL_CAPS
"""Residue names that are not one of the campaign's molecules."""


@dataclass
class RelaxReport:
    """What a finished relaxation produced for one target.

    Attributes:
        name: Target name, which is the run it carries the frame for.
        structure: The relaxed structure, holding the protein and every molecule.
        selectors: Where each molecule landed in that structure, keyed by molecule name.
    """

    name: str
    structure: Path
    selectors: dict[str, Selector] = field(default_factory=dict)

    def write(self, path: Path) -> None:
        """Write the report as JSON.

        Args:
            path: File to write to. Parent directories are created as needed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "structure": str(self.structure),
            "selectors": {
                name: selector.model_dump(exclude_none=True)
                for name, selector in self.selectors.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> RelaxReport:
        """Read a report written by a previous relaxation.

        Args:
            path: The JSON file to read.

        Returns:
            The report.

        Raises:
            InputValidationError: If the file cannot be read as a relaxation report.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                name=str(payload["name"]),
                structure=Path(str(payload["structure"])),
                selectors={
                    name: Selector.model_validate(value)
                    for name, value in payload["selectors"].items()
                },
            )
        except (OSError, ValueError, KeyError) as error:
            raise InputValidationError(
                f"{path} could not be read as a relaxation report: {error}"
            ) from error


@dataclass
class RelaxOutcome:
    """What the relaxation stage did.

    Attributes:
        targets: Names of the systems being relaxed.
        tasks: Tasks that were run or queued.
        skipped: Targets whose relaxation had already finished.
        job_id: Scheduler job identifier, when queued.
        script: The batch script, when one was written.
    """

    targets: list[str] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    job_id: str | None = None
    script: Path | None = None


def check_relaxation(request: CampaignRequest) -> None:
    """Check that a campaign's relaxation request can be satisfied.

    Args:
        request: The campaign request.

    Raises:
        OpenFEAPIError: If relaxation is enabled for a protocol that has no frame to relax.
    """
    if request.relax.enabled and not relax_targets(request):
        raise OpenFEAPIError(
            f"'relax.enabled' is set, but a '{request.protocol}' campaign has no structure "
            "to relax; remove it"
        )


def relax_targets(request: CampaignRequest) -> list[MdSystemSpec]:
    """Return the systems whose frame a relaxation would improve, as MD systems.

    A relaxation is a plain MD run, so each target is expressed as one, whatever protocol
    the campaign runs.

    Args:
        request: The campaign request.

    Returns:
        One system per frame-carrying entry, in declaration order.
    """
    if isinstance(request, AbfeRequest):
        return [
            MdSystemSpec(
                name=spec.name,
                structure=spec.structure,
                protein=spec.protein,
                ligands=[
                    MdLigandSpec(
                        name=spec.ligand_name(),
                        smiles=spec.ligand.smiles,
                        path=spec.ligand.path,
                        selector=spec.ligand.selector,
                    )
                ],
                cofactors=list(spec.cofactors),
            )
            for spec in request.complexes
        ]

    if isinstance(request, RbfeRequest | SepTopRequest):
        # Only the reference carries the frame: every other ligand is placed into it.
        reference = request.reference_ligand()
        return [
            MdSystemSpec(
                name=reference.name,
                structure=reference.structure,
                protein=request.protein,
                ligands=[
                    MdLigandSpec(
                        name=reference.name,
                        smiles=reference.smiles,
                        path=reference.path,
                        selector=reference.selector,
                    )
                ],
                cofactors=list(request.cofactors),
            )
        ]

    if isinstance(request, MdRequest):
        return list(request.systems)

    return []


CHARGE_PREFIX = "partial_charge_settings."
"""Settings the relaxation inherits from the campaign.

Only the charge settings carry over. Every other override names a path in the campaign
protocol's own settings tree, which plain MD does not have, and the charge method has to
match or the relaxation and the campaign would each pay for their own charges.
"""


def _relax_settings(relax: RelaxSpec, base: SettingsSpec) -> SettingsSpec:
    """Turn the request's relaxation settings into MD settings."""
    overrides: dict[str, object] = {
        key: value for key, value in base.overrides.items() if key.startswith(CHARGE_PREFIX)
    }
    overrides["simulation_settings.equilibration_length"] = relax.length
    overrides["simulation_settings.production_length"] = MINIMUM_PRODUCTION
    return SettingsSpec(overrides=overrides)


def _target_directory(campaign: Campaign, name: str) -> Path:
    """Return the directory holding one target's relaxation."""
    return campaign.directory / RELAXED_DIR / name


def prepare_relaxation(
    campaign: Campaign,
    check_parameters: bool = True,
    processors: int = 1,
) -> list[MdSystemSpec]:
    """Prepare and plan a relaxation for every frame-carrying system.

    A target whose relaxation has already finished is left alone, so running the stage again
    while waiting for a job costs nothing.

    Args:
        campaign: The campaign to relax.
        check_parameters: Whether to check that each prepared protein parameterizes.
        processors: Processes to use for partial charge generation.

    Returns:
        Every target of the campaign, in declaration order, whether or not it needed work.

    Raises:
        InputValidationError: If a target cannot be prepared or the protocol rejects it.
        OpenFEAPIError: If the campaign does not ask for a relaxation.
    """
    request = campaign.manifest.request
    if not request.relax.enabled:
        raise OpenFEAPIError(
            f"campaign '{campaign.manifest.name}' does not ask for relaxation; set "
            "'relax.enabled: true' in its request to use this stage"
        )

    targets = relax_targets(request)
    check_relaxation(request)
    settings = _relax_settings(request.relax, request.settings)
    finished = {task.run_name for task in relax_tasks(campaign) if task.is_complete}

    for target in targets:
        if target.name in finished:
            logger.info("The relaxation of '%s' has already finished", target.name)
            continue

        directory = _target_directory(campaign, target.name)
        report = prepare_system(target, directory / "input", check_parameters=check_parameters)
        plan_md(
            report,
            settings,
            directory,
            repeats=1,
            solvated=target.solvent,
            # Planning's own cache: charges key on molecule and method, not pose.
            charge_cache=ChargeCache(campaign.directory / "prepared" / "charges"),
            processors=processors,
        )
        logger.info("Prepared a relaxation of '%s' in %s", target.name, directory)

    return targets


def relax_tasks(campaign: Campaign) -> list[Task]:
    """Build the tasks that run a campaign's relaxations.

    Args:
        campaign: The campaign to relax.

    Returns:
        One task per prepared relaxation, in target order. Targets whose transformation has
        not been written are skipped.
    """
    tasks: list[Task] = []
    for target in relax_targets(campaign.manifest.request):
        directory = _target_directory(campaign, target.name)
        transformation = directory / f"md_{target.name}.json"
        if not transformation.is_file():
            continue
        work_dir = directory / "run"
        tasks.append(
            Task(
                run_name=target.name,
                repeat=1,
                transformation=transformation,
                work_dir=work_dir,
                result_path=work_dir / "results.json",
                cost_class=RELAX_CLASS,
                expects_estimate=False,
            )
        )
    return tasks


def _molecule_residues(structure: Structure) -> list[Selector]:
    """Return a selector for each small molecule residue, in file order."""
    return [
        Selector(chain=residue.chain_id or None, resname=residue.name, resid=int(residue.seq_id))
        for residue in structure.residues()
        if residue.name.upper() not in NON_MOLECULE_RESIDUES
    ]


def collect_relaxed(campaign: Campaign, name: str) -> RelaxReport:
    """Turn a finished relaxation into a structure preparation can read.

    The frame taken is the one after NPT equilibration, which the MD protocol writes with
    water already stripped.

    Args:
        campaign: The campaign being relaxed.
        name: The target's name.

    Returns:
        The report, also written to ``relax_report.json``.

    Raises:
        InputValidationError: If the relaxation has not finished, or its frame does not hold
            the molecules the target declared.
    """
    directory = _target_directory(campaign, name)
    report_path = directory / "relax_report.json"
    if report_path.is_file():
        return RelaxReport.read(report_path)

    result_path = directory / "run" / "results.json"
    if not result_path.is_file():
        raise InputValidationError(
            f"the relaxation of '{name}' has not finished: no result at {result_path}. Run "
            "'openfe-api relax' and wait for the job to complete."
        )

    repeat = read_md_repeat(result_path, 1)
    frame = repeat.artifacts.get("npt_structure")
    if frame is None or not frame.is_file():
        raise InputValidationError(
            f"the relaxation of '{name}' wrote no equilibrated structure"
            + (f": {repeat.failure}" if repeat.failure else "")
        )

    structure_path = directory / "system.pdb"
    shutil.copyfile(frame, structure_path)

    target = next(spec for spec in relax_targets(campaign.manifest.request) if spec.name == name)
    expected = [molecule.name for molecule in target.molecules()]
    found = _molecule_residues(Structure.load(structure_path))
    if len(found) != len(expected):
        raise InputValidationError(
            f"the relaxed structure of '{name}' holds {len(found)} molecule residue(s) but "
            f"the system declared {len(expected)}: {', '.join(expected)}. The frame is at "
            f"{structure_path}."
        )

    report = RelaxReport(
        name=name,
        structure=structure_path,
        selectors=dict(zip(expected, found, strict=True)),
    )
    report.write(report_path)
    logger.info("Collected the relaxation of '%s' into %s", name, structure_path)
    return report


def _from_frame[SpecT: MoleculeSpec](spec: SpecT, report: RelaxReport) -> SpecT:
    """Point one molecule at the residue it landed in, inside the relaxed frame."""
    name = getattr(spec, "name", None) or report.name
    return spec.model_copy(update={"path": None, "selector": report.selectors[name]})


def _from_frame_all[SpecT: MoleculeSpec](specs: list[SpecT], report: RelaxReport) -> list[SpecT]:
    """Point every molecule at the residue it landed in."""
    return [_from_frame(spec, report) for spec in specs]


def relaxed_request(campaign: Campaign) -> CampaignRequest:
    """Return the campaign's request, rewritten to read the relaxed structures.

    Molecules are read back out of the relaxed frame by the residue they landed in, and
    preparation then verifies each one against its declared SMILES.

    Args:
        campaign: The campaign being prepared.

    Returns:
        The request as given when no relaxation was asked for, otherwise a copy pointing at
        the relaxed structures.

    Raises:
        InputValidationError: If a relaxation has not finished.
    """
    request = campaign.manifest.request
    if not request.relax.enabled:
        return request

    reports = {
        target.name: collect_relaxed(campaign, target.name) for target in relax_targets(request)
    }

    if isinstance(request, AbfeRequest):
        complexes = []
        for spec in request.complexes:
            report = reports[spec.name]
            ligand = spec.ligand.model_copy(
                update={"path": None, "selector": report.selectors[spec.ligand_name()]}
            )
            complexes.append(
                spec.model_copy(
                    update={
                        "structure": report.structure,
                        "protein": spec.protein.model_copy(update={"path": None}),
                        "ligand": ligand,
                        "cofactors": _from_frame_all(spec.cofactors, report),
                        # The frame holds one copy of each molecule, so there are none to keep.
                        "extra_ligand_copies": "drop",
                    }
                )
            )
        return request.model_copy(update={"complexes": complexes})

    if isinstance(request, RbfeRequest | SepTopRequest):
        reference = request.reference_ligand()
        report = reports[reference.name]
        ligands = [
            ligand.model_copy(
                update={
                    "structure": report.structure,
                    "path": None,
                    "selector": report.selectors[reference.name],
                }
            )
            if ligand.name == reference.name
            else ligand
            for ligand in request.ligands
        ]
        return request.model_copy(
            update={
                "ligands": ligands,
                "protein": request.protein.model_copy(update={"path": None}),
                "cofactors": _from_frame_all(request.cofactors, report),
            }
        )

    if isinstance(request, MdRequest):
        systems = []
        for spec in request.systems:
            report = reports[spec.name]
            systems.append(
                spec.model_copy(
                    update={
                        "structure": report.structure,
                        "protein": (
                            spec.protein.model_copy(update={"path": None})
                            if spec.protein is not None
                            else None
                        ),
                        "ligands": _from_frame_all(spec.ligands, report),
                        "cofactors": _from_frame_all(spec.cofactors, report),
                    }
                )
            )
        return request.model_copy(update={"systems": systems})

    return request
