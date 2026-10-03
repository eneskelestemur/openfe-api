"""Pieces of the input contract shared by every protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from openfe_api.exceptions import InputValidationError

__all__ = [
    "MOLECULE_SUFFIXES",
    "NAME_PATTERN",
    "STRUCTURE_SUFFIXES",
    "CofactorSpec",
    "ExecutionSpec",
    "MoleculeSpec",
    "PathHolder",
    "ProteinSpec",
    "RelaxSpec",
    "RequestBase",
    "Selector",
    "SettingsSpec",
    "StrictModel",
    "named_paths",
    "read_request_mapping",
]

NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
"""Pattern for any name that becomes a directory or a result label."""

STRUCTURE_SUFFIXES = {".cif", ".mmcif", ".pdb"}
"""File suffixes accepted for a combined structure file."""

MOLECULE_SUFFIXES = {".sdf", ".mol", ".mol2"}
"""File suffixes accepted for a ligand or cofactor coordinate file."""

PathHolder = tuple[Any, str, str]
"""A resolvable path: the model holding it, its attribute name, and a label for errors."""


def named_paths(holders: list[PathHolder]) -> list[Path]:
    """Return the distinct input files a set of path holders names.

    Args:
        holders: Triples of holder, attribute name and label, as ``path_holders`` returns.

    Returns:
        Paths in declaration order, without duplicates.
    """
    seen: list[Path] = []
    for holder, attribute, _ in holders:
        path: Path | None = getattr(holder, attribute)
        if path is not None and path not in seen:
            seen.append(path)
    return seen


class StrictModel(BaseModel):
    """Base model that rejects unknown fields, so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class Selector(StrictModel):
    """Selects one molecule inside a structure file.

    At least one field must be set. A selector must resolve to exactly one molecule;
    resolution happens during preparation, not here.

    Attributes:
        chain: Chain identifier, e.g. ``E`` for the AF3 ligand chain.
        resname: Residue name, e.g. ``LIG_E`` (AF3) or ``LIG1`` (Boltz-2).
        resid: Residue sequence number.
    """

    chain: str | None = None
    resname: str | None = None
    resid: int | None = None

    @model_validator(mode="after")
    def _require_one_field(self) -> Selector:
        if self.chain is None and self.resname is None and self.resid is None:
            raise ValueError("a selector must set at least one of 'chain', 'resname' or 'resid'")
        return self

    def describe(self) -> str:
        """Return a human-readable form of the selector.

        Returns:
            A string such as ``chain=E resname=LIG_E``, or ``<empty>`` if nothing is set.
        """
        parts = [
            f"{key}={value}"
            for key, value in (
                ("chain", self.chain),
                ("resname", self.resname),
                ("resid", self.resid),
            )
            if value is not None
        ]
        return " ".join(parts) if parts else "<empty>"


class MoleculeSpec(StrictModel):
    """Shared fields for ligands and cofactors.

    Attributes:
        smiles: Reference chemistry for the molecule, including protonation and
            stereochemistry. Coordinates supply only the pose.
        path: Dedicated coordinate file. If omitted, the request's structure file is used.
        selector: Selects the molecule inside its source file.
    """

    smiles: str = Field(min_length=1)
    path: Path | None = None
    selector: Selector | None = None

    @field_validator("smiles")
    @classmethod
    def _reject_blank_smiles(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("'smiles' must not be blank")
        return value.strip()


class CofactorSpec(MoleculeSpec):
    """A cofactor that is present, unchanged, in both end states.

    Attributes:
        name: Cofactor name, unique within its campaign or complex.
    """

    name: str = Field(pattern=NAME_PATTERN)


class ProteinSpec(StrictModel):
    """The protein of a complex, or of an RBFE campaign's reference.

    Attributes:
        path: Dedicated protein file. If omitted, the relevant structure file is used.
        chains: Chains to keep. If omitted, every polymer chain in the source is kept.
        ph: pH used when adding hydrogens during preparation.
        keep_waters: Whether to keep crystallographic or predicted waters.
    """

    path: Path | None = None
    chains: list[str] | None = None
    ph: float = Field(default=7.4, ge=0.0, le=14.0)
    keep_waters: bool = False

    @field_validator("chains")
    @classmethod
    def _reject_empty_chain_list(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and not value:
            raise ValueError("'chains' must not be an empty list; omit it to keep every chain")
        return value


class RelaxSpec(StrictModel):
    """An optional short MD relaxation of the input structure, before preparation.

    A structure model's output carries strain a force field would not produce. Relaxing before
    placement fixes the frame for the whole campaign, which a protocol's own pre-alchemical
    equilibration cannot do: that runs per edge, after placement.

    Attributes:
        enabled: Whether to relax the input structure before preparing it.
        length: Production length of the relaxation. Short on purpose: this is meant to
            relieve strain, not to sample.
        seed: Random seed, recorded so a relaxed campaign can be reproduced.
    """

    enabled: bool = False
    length: str = "1 nanosecond"
    seed: int = Field(default=0xF00D, ge=0)


class SettingsSpec(StrictModel):
    """Protocol settings selection.

    Attributes:
        preset: Named settings preset.
        overrides: Dotted-path overrides applied on top of the preset, for example
            ``{"thermo_settings.temperature": "310 kelvin"}``.
    """

    preset: Literal["default", "screening"] = "default"
    overrides: dict[str, Any] = Field(default_factory=dict)


_PROFILE_FIELDS = ("backend", "mps", "jobs_per_gpu")


class ExecutionSpec(StrictModel):
    """How many repeats to run, and which execution profile runs them.

    Where the work runs and how repeats share a GPU are hardware properties, so they belong to
    the execution profile alone and there is exactly one place each is set.

    Attributes:
        profile: Name of the execution profile to use. The built-in ``local`` profile, one
            repeat per GPU on this machine, is used if omitted.
        repeats: Independent repeats per transformation. Each runs as its own process.
    """

    profile: str | None = None
    repeats: int = Field(default=3, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _reject_profile_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            misplaced = [name for name in _PROFILE_FIELDS if name in data]
            if misplaced:
                raise ValueError(
                    f"'{', '.join(misplaced)}' is set by the execution profile, not the "
                    "request. Remove it here, and set 'backend' and 'jobs_per_gpu' on the "
                    "profile named by 'execution.profile' in your profiles file."
                )
        return data


class RequestBase(StrictModel):
    """Fields and file handling common to every campaign request.

    Subclasses list their resolvable paths through :meth:`path_holders`, which drives both path
    resolution and provenance.

    Attributes:
        name: Campaign name, used for the campaign directory.
        settings: Protocol settings selection.
        execution: Execution backend configuration.
        relax: Optional MD relaxation of the input structure, run before preparation.
    """

    name: str = Field(pattern=NAME_PATTERN)
    settings: SettingsSpec = Field(default_factory=SettingsSpec)
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    relax: RelaxSpec = Field(default_factory=RelaxSpec)

    def initial_run_names(self) -> list[str]:
        """Return the names of the runs known from the request alone.

        A protocol whose runs are discovered during planning returns an empty list.

        Returns:
            Run names in declaration order.

        Raises:
            NotImplementedError: If a subclass does not implement it.
        """
        raise NotImplementedError

    def path_holders(self) -> list[PathHolder]:
        """Return every model attribute that holds an input path.

        Returns:
            Triples of the holding model, the attribute name, and a label naming the
            entry the path belongs to, used in error messages.

        Raises:
            NotImplementedError: If a subclass does not implement it.
        """
        raise NotImplementedError

    def resolve_paths(self, base_dir: Path) -> None:
        """Resolve input paths against a directory and check that they exist.

        Args:
            base_dir: Directory that relative paths are resolved against.

        Raises:
            InputValidationError: If a referenced input file does not exist.
        """
        base = base_dir.resolve()
        missing: list[str] = []

        for holder, attribute, label in self.path_holders():
            path: Path | None = getattr(holder, attribute)
            if path is None:
                continue
            resolved = (path if path.is_absolute() else (base / path)).resolve()
            setattr(holder, attribute, resolved)
            if not resolved.is_file():
                missing.append(f"{label}: {resolved}")

        if missing:
            listed = "\n  ".join(missing)
            raise InputValidationError(f"input files not found:\n  {listed}")

    def input_paths(self) -> list[Path]:
        """Return every input file the campaign refers to.

        Returns:
            Paths in declaration order, without duplicates.
        """
        return named_paths(self.path_holders())


def read_request_mapping(path: Path) -> dict[str, Any]:
    """Read a request YAML file into a mapping.

    Args:
        path: Path to the request YAML file.

    Returns:
        The parsed top-level mapping.

    Raises:
        InputValidationError: If the file is missing, is not valid YAML, or does not hold a
            mapping at the top level.
    """
    if not path.is_file():
        raise InputValidationError(f"request file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise InputValidationError(f"{path}: invalid YAML: {error}") from error

    if not isinstance(raw, dict):
        raise InputValidationError(f"{path}: expected a YAML mapping at the top level")

    return raw
