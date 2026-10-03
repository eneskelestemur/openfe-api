"""Tests for loading execution profiles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openfe_api.exceptions import InputValidationError
from openfe_api.execution.runner import load_profile
from openfe_api.schema.profiles import ExecutionProfile, ProfileLibrary

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_load_profile_falls_back_to_local() -> None:
    profile = load_profile(None, None)

    assert profile.name == "local"
    assert profile.backend == "local"


def test_load_profile_reads_a_file(tmp_path: Path) -> None:
    path = tmp_path / "profiles.yaml"
    path.write_text(
        json.dumps({"cluster": {"backend": "slurm", "slurm": {"partition": "gpu"}}}),
        encoding="utf-8",
    )

    profile = load_profile("cluster", path)

    assert profile.slurm is not None
    assert profile.slurm.partition == "gpu"


def test_load_profile_rejects_an_unknown_name(tmp_path: Path) -> None:
    path = tmp_path / "profiles.yaml"
    path.write_text("cluster: {backend: local}\n", encoding="utf-8")

    with pytest.raises(InputValidationError, match="unknown execution profile 'absent'"):
        load_profile("absent", path)


def test_load_profile_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="profiles file not found"):
        load_profile("cluster", tmp_path / "absent.yaml")


def test_slurm_profile_requires_a_slurm_section() -> None:
    with pytest.raises(ValueError, match="a 'slurm' section is required"):
        ExecutionProfile(name="bad", backend="slurm")


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        ExecutionProfile(name="bad", backend="grid")


def test_example_profiles_load() -> None:
    library = ProfileLibrary.from_yaml(EXAMPLES / "profiles.yaml")

    assert "cluster_gpu" in library.profiles
    assert library.get("cluster_gpu").jobs_per_gpu == 3


def test_an_unquoted_wall_time_is_rejected_with_an_explanation(tmp_path: Path) -> None:
    """YAML reads an unquoted HH:MM:SS as a base-60 number, which is easy to write by accident."""
    path = tmp_path / "profiles.yaml"
    path.write_text(
        "gpu:\n  backend: slurm\n  slurm:\n    partition: p\n    time: 4:00:00\n",
        encoding="utf-8",
    )

    with pytest.raises(InputValidationError, match="a wall time must be quoted"):
        ProfileLibrary.from_yaml(path)


def test_an_unquoted_per_class_wall_time_is_rejected_too(tmp_path: Path) -> None:
    path = tmp_path / "profiles.yaml"
    path.write_text(
        "gpu:\n  backend: slurm\n  slurm:\n    partition: p\n"
        "    time_by_class:\n      charge: 4:00:00\n",
        encoding="utf-8",
    )

    with pytest.raises(InputValidationError, match="a wall time must be quoted"):
        ProfileLibrary.from_yaml(path)
