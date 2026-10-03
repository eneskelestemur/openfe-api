"""Tests guarding package-level invariants."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

import openfe_api

PACKAGE_DIR = Path(openfe_api.__file__).parent
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_no_module_shadows_a_standard_library_module() -> None:
    shadowed = {
        path.stem for path in PACKAGE_DIR.rglob("*.py") if path.stem in sys.stdlib_module_names
    }

    assert not shadowed, f"modules shadow the standard library: {sorted(shadowed)}"


def test_stdlib_imports_survive_the_package_dir_on_sys_path() -> None:
    script = (
        "import sys, logging;"
        f"sys.path.insert(0, {str(PACKAGE_DIR)!r});"
        "import importlib; importlib.reload(logging);"
        "assert logging.getLogger is not None"
    )

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_the_citation_version_matches_the_package_version() -> None:
    """A stale CITATION.cff would have people cite a version that never existed."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))

    assert citation["version"] == pyproject["project"]["version"]
