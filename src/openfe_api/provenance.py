"""Provenance: the tool versions behind an artifact."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version

__all__ = ["collect_provenance"]

_TRACKED_PACKAGES = ("openfe-api", "openfe", "gufe", "openmm", "rdkit", "pydantic")


def collect_provenance() -> dict[str, str]:
    """Collect the versions of the tools that produce campaign artifacts.

    Packages that are not installed are reported as ``not installed`` rather than
    omitted, so a manifest records what was missing as well as what was present.

    Returns:
        A mapping of ``python`` and each tracked package name to its version string.
    """
    versions = {"python": platform.python_version()}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions
