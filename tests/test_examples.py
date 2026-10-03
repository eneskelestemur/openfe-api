"""Tests that the shipped example requests stay valid against the real input files."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from openfe_api.schema.abfe import AbfeRequest
from openfe_api.schema.md import MdRequest
from openfe_api.schema.request import load_request, validate_request

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _is_request(path: Path) -> bool:
    """Whether an example file describes a campaign request.

    Args:
        path: The YAML file to inspect.

    Returns:
        True if the file has a top-level ``protocol`` key.
    """
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    return isinstance(content, dict) and "protocol" in content


EXAMPLE_FILES = sorted(path for path in EXAMPLES_DIR.glob("*.yaml") if _is_request(path))


@pytest.mark.parametrize("example", EXAMPLE_FILES, ids=lambda path: path.stem)
def test_example_request_satisfies_the_schema(example: Path) -> None:
    """Templates point at files the repository does not ship, so only the shape is checked.

    Skipping these on a missing file would leave every template unchecked, and a typo in one
    is a bug a user hits before anything else.
    """
    request = validate_request(yaml.safe_load(example.read_text(encoding="utf-8")))

    match request:
        case AbfeRequest():
            assert request.complexes
        case MdRequest():
            assert request.systems
        case _:
            assert request.ligands


@pytest.mark.parametrize("example", EXAMPLE_FILES, ids=lambda path: path.stem)
def test_example_request_resolves_its_files(example: Path) -> None:
    try:
        request = load_request(example)
    except Exception as error:
        if "input files not found" in str(error):
            pytest.skip(f"example input data is not present: {example.name}")
        raise

    assert request.input_paths()
