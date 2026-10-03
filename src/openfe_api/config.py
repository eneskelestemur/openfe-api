"""Service configuration, read from the environment.

Every setting can be given as an environment variable prefixed with ``OPENFE_API_``, which
is how the container is configured.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["ServiceSettings", "get_settings"]


class ServiceSettings(BaseSettings):
    """How the service behaves and where it keeps its work.

    Attributes:
        root: Directory holding one subdirectory per campaign.
        profiles_file: YAML file of execution profiles. The built-in local profile is
            used if this is unset.
        processors: Processes used for partial charge generation during planning.
        check_parameters: Whether preparation runs the protein force field check.
        max_upload_bytes: Largest total size one campaign upload may carry.
    """

    model_config = SettingsConfigDict(env_prefix="OPENFE_API_", extra="ignore")

    root: Path = Field(default=Path("campaigns"))
    profiles_file: Path | None = None
    processors: int = Field(default=1, ge=1)
    check_parameters: bool = True
    max_upload_bytes: int = Field(default=100 * 1024 * 1024, ge=1)

    def campaign_dir(self, name: str) -> Path:
        """Return the directory a named campaign lives in.

        Args:
            name: Campaign name.

        Returns:
            The campaign directory, which may not exist yet.
        """
        return self.root / name


def get_settings() -> ServiceSettings:
    """Build the service settings from the environment.

    Returns:
        The settings.
    """
    return ServiceSettings()
