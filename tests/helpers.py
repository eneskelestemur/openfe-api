"""Builders for the inputs the tests run against.

The result files are written with the same gufe JSON codec that ``openfe quickrun`` uses and
follow the structure the OpenFE 1.12 analysis units produce. The campaign builders make a
planned campaign without doing any real work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from gufe.tokenization import JSON_HANDLER
from openff.units import unit

from openfe_api.campaign import Campaign, RunState
from openfe_api.execution.base import Task
from openfe_api.schema.profiles import ExecutionProfile, SlurmProfile
from openfe_api.schema.request import validate_request

DATA = Path(__file__).parent / "data"

__all__ = [
    "KCAL",
    "LOCAL_PROFILE",
    "SLURM_PROFILE",
    "SUCCESSFUL_RESULT",
    "abfe_campaign",
    "leg_outputs",
    "mini_series_data",
    "network_campaign",
    "overlap_matrix",
    "write_md_result",
    "write_phase_result",
    "write_result",
    "write_result_file",
    "write_septop_result",
]

KCAL = unit.kilocalorie_per_mole


def overlap_matrix(size: int = 4, neighbor: float = 0.3) -> np.ndarray:
    """Build an overlap matrix with a given neighboring value.

    Args:
        size: Matrix dimension.
        neighbor: Value to place next to the diagonal.

    Returns:
        The matrix.
    """
    matrix = np.eye(size) * (1 - 2 * neighbor)
    for index in range(size - 1):
        matrix[index, index + 1] = neighbor
        matrix[index + 1, index] = neighbor
    return matrix


def leg_outputs(
    simtype: str,
    estimate: float,
    error: float = 0.2,
    neighbor_overlap: float = 0.3,
    exchange: float = 0.2,
    converged: bool = True,
) -> dict[str, Any]:
    """Build the outputs an analysis unit reports for one leg.

    Args:
        simtype: ``complex`` or ``solvent``.
        estimate: Leg estimate in kcal/mol.
        error: Leg MBAR error in kcal/mol.
        neighbor_overlap: Overlap between neighboring lambda states.
        exchange: Replica exchange probability between neighboring states.
        converged: Whether forward and reverse estimates should agree.

    Returns:
        The outputs dictionary.
    """
    fractions = np.array([0.25, 0.5, 0.75, 1.0])
    forward = np.array([estimate] * 4)
    reverse = forward if converged else forward + 5.0
    return {
        "simtype": simtype,
        "unit_estimate": estimate * KCAL,
        "unit_estimate_error": error * KCAL,
        "standard_state_correction": (-1.2 if simtype == "complex" else 0.0) * KCAL,
        "production_iterations": 4000,
        "unit_mbar_overlap": {"matrix": overlap_matrix(neighbor=neighbor_overlap)},
        "replica_exchange_statistics": {"matrix": overlap_matrix(neighbor=exchange)},
        "forward_and_reverse_energies": {
            "fractions": fractions,
            "forward_DGs": forward * KCAL,
            "forward_dDGs": np.array([error] * 4) * KCAL,
            "reverse_DGs": reverse * KCAL,
            "reverse_dDGs": np.array([error] * 4) * KCAL,
        },
    }


def write_result(
    path: Path,
    ligand: str = "mini",
    estimate: float = -8.0,
    uncertainty: float = 0.3,
    legs: dict[str, dict[str, Any]] | None = None,
    failed: bool = False,
    no_estimate: bool = False,
) -> Path:
    """Write a synthetic result file.

    Args:
        path: File to write.
        ligand: Ligand name to record.
        estimate: Overall estimate in kcal/mol.
        uncertainty: Overall uncertainty in kcal/mol.
        legs: Leg outputs keyed by simulation type. Defaults to one converged pair.
        failed: Whether every unit should carry an exception.
        no_estimate: Whether to omit the overall estimate.

    Returns:
        The path written.
    """
    if legs is None:
        legs = {
            "complex": leg_outputs("complex", -12.0),
            "solvent": leg_outputs("solvent", -4.0),
        }

    unit_results: dict[str, Any] = {}
    if failed:
        unit_results["ProtocolUnitResult-0"] = {
            "source_key": "ABFEComplexSimUnit-abc",
            "exception": ["RuntimeError", ["particle position is NaN"]],
            "outputs": {},
        }
    else:
        unit_results["ProtocolUnitResult-setup"] = {
            "source_key": "ABFEComplexSetupUnit-abc",
            "outputs": {"simtype": "complex"},
        }
        for index, (simtype, outputs) in enumerate(legs.items()):
            unit_results[f"ProtocolUnitResult-{index}"] = {
                "source_key": f"ABFE{simtype.capitalize()}AnalysisUnit-abc",
                "outputs": outputs,
            }

    payload: dict[str, Any] = {
        "estimate": None if (no_estimate or failed) else estimate * KCAL,
        "uncertainty": None if (no_estimate or failed) else uncertainty * KCAL,
        "unit_results": unit_results,
        "protocol_result": {
            "data": {
                "solvent": {
                    "unit-key": [
                        {
                            "inputs": {
                                "setup_results": {
                                    "inputs": {
                                        "alchemical_components": {
                                            "stateA": [{"molprops": {"ofe-name": ligand}}]
                                        }
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, cls=JSON_HANDLER.encoder), encoding="utf-8")
    return path


def write_phase_result(
    path: Path,
    estimate: float,
    uncertainty: float = 0.2,
    reported_uncertainty: float | None = None,
    neighbor_overlap: float = 0.3,
    exchange: float = 0.2,
    converged: bool = True,
    failed: bool = False,
    no_estimate: bool = False,
) -> Path:
    """Write a synthetic result file for one RBFE phase.

    An RBFE transformation covers a single phase, so the file carries one estimate rather
    than a leg each for complex and solvent.

    Args:
        path: File to write.
        estimate: Phase estimate in kcal/mol.
        uncertainty: The analysis unit's MBAR error in kcal/mol.
        reported_uncertainty: The file's top-level uncertainty, which OpenFE computes as the
            spread across its own repeats and so reports as zero for a single repeat.
            Defaults to ``uncertainty``.
        neighbor_overlap: Overlap between neighboring lambda states.
        exchange: Replica exchange probability between neighboring states.
        converged: Whether forward and reverse estimates should agree.
        failed: Whether every unit should carry an exception.
        no_estimate: Whether to omit the estimate.

    Returns:
        The path written.
    """
    fractions = np.array([0.25, 0.5, 0.75, 1.0])
    forward = np.array([estimate] * 4)
    reverse = forward if converged else forward + 5.0

    unit_results: dict[str, Any] = {}
    if failed:
        unit_results["ProtocolUnitResult-0"] = {
            "source_key": "HybridTopologyMultiStateSimulationUnit-abc",
            "exception": ["RuntimeError", ["particle position is NaN"]],
            "outputs": {},
        }
    else:
        unit_results["ProtocolUnitResult-0"] = {
            "source_key": "HybridTopologyMultiStateAnalysisUnit-abc",
            "outputs": {
                "unit_estimate": estimate * KCAL,
                "unit_estimate_error": uncertainty * KCAL,
                "production_iterations": 4000,
                "unit_mbar_overlap": {"matrix": overlap_matrix(neighbor=neighbor_overlap)},
                "replica_exchange_statistics": {"matrix": overlap_matrix(neighbor=exchange)},
                "forward_and_reverse_energies": {
                    "fractions": fractions,
                    "forward_DGs": forward * KCAL,
                    "forward_dDGs": np.array([uncertainty] * 4) * KCAL,
                    "reverse_DGs": reverse * KCAL,
                    "reverse_dDGs": np.array([uncertainty] * 4) * KCAL,
                },
            },
        }

    top_level = uncertainty if reported_uncertainty is None else reported_uncertainty
    payload: dict[str, Any] = {
        "estimate": None if (no_estimate or failed) else estimate * KCAL,
        "uncertainty": None if (no_estimate or failed) else top_level * KCAL,
        "unit_results": unit_results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, cls=JSON_HANDLER.encoder), encoding="utf-8")
    return path


def write_septop_result(
    path: Path,
    ddg: float,
    complex_estimate: float = -10.0,
    uncertainty: float = 0.2,
    neighbor_overlap: float = 0.3,
    exchange: float = 0.2,
    converged: bool = True,
    failed: bool = False,
    no_estimate: bool = False,
) -> Path:
    """Write a synthetic result file for one repeat of one SepTop edge.

    A SepTop transformation covers both phases, so one file carries an analysis unit for each
    and a top-level estimate that is already the relative binding free energy. The top-level
    uncertainty is the spread across the protocol's own repeats, which is one here, so it is
    written as zero exactly as OpenFE writes it.

    Args:
        path: File to write.
        ddg: The relative binding free energy in kcal/mol.
        complex_estimate: The complex phase estimate in kcal/mol; the solvent phase is set so
            the two differ by ``ddg`` once the standard state corrections are applied.
        uncertainty: Each phase's MBAR error in kcal/mol.
        neighbor_overlap: Overlap between neighboring lambda states.
        exchange: Replica exchange probability between neighboring states.
        converged: Whether forward and reverse estimates should agree.
        failed: Whether every unit should carry an exception.
        no_estimate: Whether to omit the estimate.

    Returns:
        The path written.
    """
    corrections = {
        "standard_state_correction_A": -1.1 * KCAL,
        "standard_state_correction_B": 0.9 * KCAL,
    }
    complex_outputs = leg_outputs(
        "complex",
        complex_estimate,
        error=uncertainty,
        neighbor_overlap=neighbor_overlap,
        exchange=exchange,
        converged=converged,
    )
    del complex_outputs["standard_state_correction"]
    complex_outputs.update(corrections)
    solvent_outputs = leg_outputs(
        "solvent",
        complex_estimate - ddg,
        error=uncertainty,
        neighbor_overlap=neighbor_overlap,
        exchange=exchange,
        converged=converged,
    )

    unit_results: dict[str, Any] = {}
    if failed:
        unit_results["ProtocolUnitResult-0"] = {
            "source_key": "SepTopComplexRunUnit-abc",
            "exception": ["RuntimeError", ["particle position is NaN"]],
            "outputs": {},
        }
    else:
        unit_results["ProtocolUnitResult-setup"] = {
            "source_key": "SepTopComplexSetupUnit-abc",
            "outputs": {"simtype": "complex"},
        }
        for index, (phase, outputs) in enumerate(
            (("solvent", solvent_outputs), ("complex", complex_outputs))
        ):
            unit_results[f"ProtocolUnitResult-{index}"] = {
                "source_key": f"SepTop{phase.capitalize()}AnalysisUnit-abc",
                "outputs": outputs,
            }

    payload: dict[str, Any] = {
        "estimate": None if (no_estimate or failed) else ddg * KCAL,
        "uncertainty": None if (no_estimate or failed) else 0.0 * KCAL,
        "unit_results": unit_results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, cls=JSON_HANDLER.encoder), encoding="utf-8")
    return path


def write_md_result(
    path: Path,
    failed: bool = False,
    no_trajectory: bool = False,
) -> Path:
    """Write a synthetic result file for one repeat of a plain MD run.

    A finished MD run writes a null estimate, because ``PlainMDProtocolResult.get_estimate``
    returns None by design, so the file is written that way here too: what marks it finished
    is the trajectory its simulation unit reported.

    Args:
        path: File to write.
        failed: Whether every unit should carry an exception.
        no_trajectory: Whether to omit the trajectory output, as an interrupted run would.

    Returns:
        The path written.
    """
    shared = path.parent / "shared_PlainMDSimulationUnit-abc_attempt_0"
    unit_results: dict[str, Any] = {
        "ProtocolUnitResult-setup": {
            "source_key": "PlainMDSetupUnit-abc",
            "outputs": {"system": str(shared / "system.xml.bz2")},
        }
    }
    if failed:
        unit_results["ProtocolUnitResult-0"] = {
            "source_key": "PlainMDSimulationUnit-abc",
            "exception": ["RuntimeError", ["particle position is NaN"]],
            "outputs": {},
        }
    else:
        outputs: dict[str, Any] = {
            "repeat_id": 1,
            "generation": 0,
            "system_pdb": str(shared / "system.pdb"),
            "minimized_pdb": str(shared / "minimized.pdb"),
            "nvt_equil_pdb": str(shared / "equil_nvt.pdb"),
            "npt_equil_pdb": str(shared / "equil_npt.pdb"),
            "last_checkpoint": None,
        }
        if not no_trajectory:
            outputs["nc"] = str(shared / "simulation.xtc")
        unit_results["ProtocolUnitResult-0"] = {
            "source_key": "PlainMDSimulationUnit-abc",
            "outputs": outputs,
        }

    payload: dict[str, Any] = {
        "estimate": None,
        "uncertainty": None,
        "unit_results": unit_results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, cls=JSON_HANDLER.encoder), encoding="utf-8")
    return path


SUCCESSFUL_RESULT = json.dumps(
    {
        "estimate": {"magnitude": -8.0},
        "uncertainty": {"magnitude": 0.2},
        "unit_results": {"ProtocolUnitResult-0": {"source_key": "unit", "outputs": {}}},
    }
)
"""The shape of a result file that records a finished run."""

SLURM_PROFILE = ExecutionProfile(
    name="test_slurm",
    backend="slurm",
    jobs_per_gpu=3,
    setup_commands=["module load cuda"],
    slurm=SlurmProfile(partition="gpu", qos="gpu_access", gres="gpu:1"),
)
LOCAL_PROFILE = ExecutionProfile(name="test_local", backend="local", gpus=[0])


def abfe_campaign(tmp_path: Path, repeats: int = 3, planned: bool = True) -> Campaign:
    """Create a campaign whose runs are planned, without doing real work.

    Args:
        tmp_path: Pytest temporary directory.
        repeats: Repeats per run.
        planned: Whether to move the run into the planned state and write a plan file.

    Returns:
        The campaign.
    """
    request = validate_request(
        {
            "protocol": "abfe",
            "name": "exec_campaign",
            "complexes": [
                {
                    "name": "mini",
                    "structure": str(DATA / "mini_complex.cif"),
                    "protein": {"chains": ["A"]},
                    "ligand": {"selector": {"chain": "L"}, "smiles": "C[C@H](O)c1ccccc1"},
                    "extra_ligand_copies": "drop",
                }
            ],
            "execution": {"repeats": repeats},
        }
    )
    campaign = Campaign.create(tmp_path / "campaign", request)
    if planned:
        plan = campaign.directory / "plans" / "abfe_mini.json"
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text("{}", encoding="utf-8")
        campaign.set_state("mini", RunState.PREPARED)
        campaign.set_state("mini", RunState.PLANNED)
    return campaign


def write_result_file(task: Task, estimate: float | None) -> None:
    """Write a result file for a task, successful or failed.

    Args:
        task: The task to write results for.
        estimate: The estimate to record, or None for a failed run.
    """
    task.work_dir.mkdir(parents=True, exist_ok=True)
    units: dict[str, Any] = (
        {"ProtocolUnitResult-0": {"source_key": "ABFEComplexSimUnit-a", "outputs": {}}}
        if estimate is not None
        else {
            "ProtocolUnitResult-0": {
                "source_key": "ABFEComplexSimUnit-a",
                "exception": ["OpenMMException", ["No compatible CUDA device is available"]],
            }
        }
    )
    payload = {
        "estimate": {"magnitude": estimate} if estimate is not None else None,
        "uncertainty": {"magnitude": 0.2} if estimate is not None else None,
        "unit_results": units,
    }
    task.result_path.write_text(json.dumps(payload), encoding="utf-8")


MINI_REFERENCE = "C[C@H](O)c1ccccc1"
"""SMILES of the ligand in the trimmed complex fixture."""

MINI_ANALOGUE = "C[C@H](O)c1ccc(F)cc1"
"""A close analogue of it, for the smallest series that has an edge."""


def mini_series_data(
    protocol: str,
    name: str,
    ligands: tuple[str, str] = ("reference", "fluoro"),
    **overrides: Any,
) -> dict[str, Any]:
    """Build request data for a two-ligand series on the trimmed complex fixture.

    Args:
        protocol: The protocol to name, ``rbfe`` or ``septop``.
        name: Campaign name.
        ligands: Names for the reference and its analogue.
        **overrides: Further request fields, such as ``settings`` or ``execution``.

    Returns:
        A request dictionary ready for validation.
    """
    reference, analogue = ligands
    data: dict[str, Any] = {
        "protocol": protocol,
        "name": name,
        "ligands": [
            {
                "name": reference,
                "structure": str(DATA / "mini_complex.cif"),
                "selector": {"chain": "L"},
                "smiles": MINI_REFERENCE,
            },
            {"name": analogue, "smiles": MINI_ANALOGUE},
        ],
    }
    data.update(overrides)
    return data


def network_campaign(
    tmp_path: Path, protocol: str, summary: dict[str, Any], edge: str = "a_to_b"
) -> Campaign:
    """Create a network campaign whose single edge has been submitted.

    Args:
        tmp_path: Pytest temporary directory.
        protocol: The protocol to name, ``rbfe`` or ``septop``.
        summary: The ``network.json`` summary planning would have written.
        edge: The edge's name.

    Returns:
        The campaign.
    """
    request = validate_request(
        mini_series_data(
            protocol, f"{protocol}_results", ligands=("a", "b"), execution={"repeats": 1}
        )
    )
    campaign = Campaign.create(tmp_path / "campaign", request)
    plans = campaign.directory / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    (plans / "network.json").write_text(json.dumps(summary), encoding="utf-8")
    campaign.add_runs([edge], repeats=1, state=RunState.PREPARED)
    campaign.set_state(edge, RunState.PLANNED)
    campaign.set_state(edge, RunState.SUBMITTED)
    return campaign
