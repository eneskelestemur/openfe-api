"""Tests for the pieces of the contract every protocol shares."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from openfe_api.exceptions import InputValidationError
from openfe_api.schema.common import Selector
from openfe_api.schema.request import validate_request


def test_empty_selector_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one of"):
        Selector()


def test_selector_describe() -> None:
    assert Selector(chain="E", resname="LIG_E").describe() == "chain=E resname=LIG_E"


@pytest.mark.parametrize(
    "misplaced",
    [
        {"backend": "slurm"},
        {"mps": {"enabled": True, "jobs_per_gpu": 3}},
        {"jobs_per_gpu": 3},
    ],
    ids=["backend", "mps", "jobs_per_gpu"],
)
def test_profile_settings_are_rejected_in_the_request(
    split_request_data: dict[str, Any], misplaced: dict[str, Any]
) -> None:
    """GPU packing and the backend live on the profile only, so they cannot be ignored."""
    split_request_data["execution"] = {"repeats": 3, **misplaced}

    with pytest.raises(InputValidationError, match="is set by the execution profile"):
        validate_request(split_request_data)


def test_execution_defaults_to_the_local_profile(split_request_data: dict[str, Any]) -> None:
    request = validate_request(split_request_data)

    assert request.execution.profile is None
    assert request.execution.repeats == 3


def test_empty_chain_list_is_rejected(split_request_data: dict[str, Any]) -> None:
    split_request_data["complexes"][0]["protein"]["chains"] = []

    with pytest.raises(InputValidationError, match="must not be an empty list"):
        validate_request(split_request_data)
