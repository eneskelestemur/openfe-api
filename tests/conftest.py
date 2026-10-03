"""Shared fixtures for the openfe-api test suite."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

LIGAND_SMILES = "Cc1ccc(S(=O)(=O)c2c(N)n(C(O)c3ccc([N+](=O)[O-])cc3)c3nc4ccccc4nc23)cc1"
COFACTOR_SMILES = "NC(=O)c1ccccc1"

REFERENCE_SMILES = "Cc1ccc(cc1)C(=O)Nc1ccccc1"
ANALOGUE_SMILES = "CCc1ccc(cc1)C(=O)Nc1ccccc1"
THIRD_SMILES = "Clc1ccc(cc1)C(=O)Nc1ccccc1"
HOP_SMILES = "O=C(Nc1ccncc1)C1CCOCC1"


@pytest.fixture
def input_dir(tmp_path: Path) -> Path:
    """Create a directory holding placeholder input files.

    The files only need to exist: the request schema checks paths, never contents.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Directory containing ``model.cif``, ``protein.pdb``, ``ligand.sdf`` and
        ``cofactor.sdf``.
    """
    directory = tmp_path / "inputs"
    directory.mkdir()
    for name in ("model.cif", "protein.pdb", "ligand.sdf", "cofactor.sdf"):
        (directory / name).write_text("placeholder\n", encoding="utf-8")
    return directory


@pytest.fixture
def combined_request_data(input_dir: Path) -> dict[str, Any]:
    """Return request data for a ligand read from a combined structure file.

    Args:
        input_dir: Directory holding the placeholder input files.

    Returns:
        A request dictionary ready for validation.
    """
    return {
        "protocol": "abfe",
        "name": "combined_campaign",
        "complexes": [
            {
                "name": "complex_one",
                "structure": str(input_dir / "model.cif"),
                "protein": {"chains": ["A", "B"]},
                "ligand": {
                    "selector": {"chain": "E", "resname": "LIG_E"},
                    "smiles": LIGAND_SMILES,
                },
                "extra_ligand_copies": "drop",
                "cofactors": [
                    {
                        "name": "NAD_C",
                        "selector": {"chain": "C"},
                        "smiles": COFACTOR_SMILES,
                    }
                ],
            }
        ],
        "execution": {"repeats": 2},
    }


@pytest.fixture
def split_request_data(input_dir: Path) -> dict[str, Any]:
    """Return request data for components supplied as separate files.

    Args:
        input_dir: Directory holding the placeholder input files.

    Returns:
        A request dictionary ready for validation.
    """
    return {
        "protocol": "abfe",
        "name": "split_campaign",
        "complexes": [
            {
                "name": "complex_one",
                "protein": {"path": str(input_dir / "protein.pdb")},
                "ligand": {"path": str(input_dir / "ligand.sdf"), "smiles": LIGAND_SMILES},
            }
        ],
    }


@pytest.fixture
def write_request(tmp_path: Path) -> Callable[..., Path]:
    """Return a helper that writes request data to a YAML file.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        A callable taking request data and an optional file name, returning the
        path it was written to.
    """

    def _write(data: dict[str, Any], name: str = "request.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def series_request_data(input_dir: Path) -> dict[str, Any]:
    """Return request data for a three-ligand series in a predicted complex.

    The reference carries the protein in a combined structure file; the other two ligands
    are SMILES only, so their poses are generated from the common core.

    Args:
        input_dir: Directory holding the placeholder input files.

    Returns:
        A request dictionary ready for validation.
    """
    return {
        "protocol": "rbfe",
        "name": "series_campaign",
        "protein": {"chains": ["A"]},
        "ligands": [
            {
                "name": "lig_ref",
                "smiles": REFERENCE_SMILES,
                "structure": str(input_dir / "model.cif"),
                "selector": {"chain": "B", "resname": "LIG"},
            },
            {"name": "lig_b", "smiles": ANALOGUE_SMILES},
            {"name": "lig_c", "smiles": THIRD_SMILES},
        ],
    }


@pytest.fixture
def septop_request_data(input_dir: Path) -> dict[str, Any]:
    """Return request data for a SepTop campaign holding one analogue and one scaffold hop.

    Args:
        input_dir: Directory holding the placeholder input files.

    Returns:
        A request dictionary ready for validation.
    """
    return {
        "protocol": "septop",
        "name": "septop_campaign",
        "protein": {"chains": ["A"]},
        "ligands": [
            {
                "name": "lig_ref",
                "smiles": REFERENCE_SMILES,
                "structure": str(input_dir / "model.cif"),
                "selector": {"chain": "B", "resname": "LIG"},
            },
            {"name": "lig_b", "smiles": ANALOGUE_SMILES},
            {"name": "lig_hop", "smiles": HOP_SMILES},
        ],
    }


@pytest.fixture
def md_request_data(input_dir: Path) -> dict[str, Any]:
    """Return request data for a plain MD run of one solvated complex.

    Args:
        input_dir: Directory holding the placeholder input files.

    Returns:
        A request dictionary ready for validation.
    """
    return {
        "protocol": "md",
        "name": "md_campaign",
        "systems": [
            {
                "name": "complex_one",
                "structure": str(input_dir / "model.cif"),
                "protein": {"chains": ["A"]},
                "ligands": [
                    {
                        "name": "lig",
                        "selector": {"chain": "E", "resname": "LIG_E"},
                        "smiles": LIGAND_SMILES,
                    }
                ],
            }
        ],
        "execution": {"repeats": 1},
    }
