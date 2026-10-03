"""Execution profiles: the cluster-specific half of running a campaign.

A profile holds everything that depends on where the work runs -- Slurm partitions, wall
time, how the environment is activated -- so a request stays portable between a laptop, a
server VM and an HPC cluster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from openfe_api.exceptions import InputValidationError

__all__ = ["ExecutionProfile", "ProfileLibrary", "SlurmProfile"]


class SlurmProfile(BaseModel):
    """Slurm submission settings.

    Attributes:
        partition: Partition(s) to submit to, as written after ``--partition``.
        qos: Quality of service, if the cluster requires one.
        account: Account to charge, if the cluster requires one.
        time: Wall time limit, as ``D-HH:MM:SS`` or ``HH:MM:SS``. Quote it in YAML: an
            unquoted ``HH:MM:SS`` is read as a base-60 number.
        time_by_class: Wall time per cost class, overriding ``time`` for that class. A
            charge-corrected RBFE edge runs 22 lambda windows for 20 ns each, roughly four
            times a neutral edge, so give the ``charge`` class its own limit rather than
            letting it inherit a limit sized for neutral edges.
        gres: Generic resource request, normally the GPU request.
        cpus_per_task: CPU cores per job.
        memory: Memory request, as written after ``--mem``.
        extra_directives: Further ``#SBATCH`` lines, without the ``#SBATCH`` prefix.
    """

    model_config = ConfigDict(extra="forbid")

    partition: str
    qos: str | None = None
    account: str | None = None
    time: str = "3-00:00:00"
    time_by_class: dict[str, str] = Field(default_factory=dict)

    @field_validator("time", "time_by_class", mode="before")
    @classmethod
    def _reject_unquoted_time(cls, value: Any) -> Any:
        # YAML reads 4:00:00 as base 60, so a wall time can arrive as an int.
        values = value.values() if isinstance(value, dict) else [value]
        for entry in values:
            if isinstance(entry, int | float):
                raise ValueError(
                    f"a wall time must be quoted: YAML reads {entry} where you wrote a time "
                    "like 4:00:00, because an unquoted HH:MM:SS is a base-60 number. Write "
                    "'4:00:00' or 0-04:00:00 instead."
                )
        return value

    gres: str | None = "gpu:1"
    cpus_per_task: int = Field(default=8, ge=1)
    memory: str = "64G"
    extra_directives: list[str] = Field(default_factory=list)


class ExecutionProfile(BaseModel):
    """How and where a campaign's simulations run.

    Attributes:
        name: Profile name, referenced by a request's ``execution.profile``.
        backend: Which backend this profile configures.
        setup_commands: Shell lines run before the simulation, normally activating the
            environment holding OpenFE.
        gpus: GPU device indices available to the local backend. If None, the backend
            detects them, falling back to a single device.
        jobs_per_gpu: Concurrent repeats per GPU. Above one, CUDA MPS is used.
        slurm: Slurm settings, required when ``backend`` is ``slurm``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    backend: str = "local"
    setup_commands: list[str] = Field(default_factory=list)
    gpus: list[int] | None = None
    jobs_per_gpu: int = Field(default=1, ge=1)
    slurm: SlurmProfile | None = None

    @model_validator(mode="after")
    def _check_backend(self) -> ExecutionProfile:
        if self.backend not in {"local", "slurm"}:
            raise ValueError(f"unknown backend '{self.backend}'; use 'local' or 'slurm'")
        if self.backend == "slurm" and self.slurm is None:
            raise ValueError(f"profile '{self.name}': a 'slurm' section is required")
        return self


class ProfileLibrary:
    """A collection of named execution profiles.

    Attributes:
        profiles: The profiles, keyed by name.
        source: Where the profiles were loaded from, for error messages.
    """

    def __init__(self, profiles: dict[str, ExecutionProfile], source: Path | None = None) -> None:
        """Initialize a library.

        Args:
            profiles: Profiles keyed by name.
            source: File the profiles came from, if any.
        """
        self.profiles = profiles
        self.source = source

    @classmethod
    def default(cls) -> ProfileLibrary:
        """Return the built-in library, holding a single local profile.

        Returns:
            A library with one profile named ``local``.
        """
        return cls({"local": ExecutionProfile(name="local", backend="local")})

    @classmethod
    def from_yaml(cls, path: Path) -> ProfileLibrary:
        """Load profiles from a YAML file.

        The file maps profile names to profile bodies. The built-in ``local`` profile is
        included unless the file defines one with that name.

        Args:
            path: The YAML file to load.

        Returns:
            The loaded library.

        Raises:
            InputValidationError: If the file is missing, malformed, or holds an invalid
                profile.
        """
        if not path.is_file():
            raise InputValidationError(f"profiles file not found: {path}")

        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise InputValidationError(f"{path}: invalid YAML: {error}") from error

        if not isinstance(raw, dict):
            raise InputValidationError(f"{path}: expected a mapping of profile names to profiles")

        profiles = dict(cls.default().profiles)
        for name, body in raw.items():
            if not isinstance(body, dict):
                raise InputValidationError(f"{path}: profile '{name}' must be a mapping")
            data: dict[str, Any] = {"name": name, **body}
            try:
                profiles[name] = ExecutionProfile.model_validate(data)
            except ValueError as error:
                raise InputValidationError(f"{path}: profile '{name}': {error}") from error

        return cls(profiles, source=path)

    def get(self, name: str) -> ExecutionProfile:
        """Return a profile by name.

        Args:
            name: The profile name.

        Returns:
            The profile.

        Raises:
            InputValidationError: If no profile has that name.
        """
        try:
            return self.profiles[name]
        except KeyError:
            where = f" in {self.source}" if self.source else ""
            raise InputValidationError(
                f"unknown execution profile '{name}'{where}; available: "
                f"{', '.join(sorted(self.profiles))}"
            ) from None
