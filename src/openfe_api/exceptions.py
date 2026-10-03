"""Exception types raised across ``openfe_api``."""

from __future__ import annotations

__all__ = [
    "CampaignStateError",
    "InputValidationError",
    "OpenFEAPIError",
]


class OpenFEAPIError(Exception):
    """Base class for all errors raised by ``openfe_api``."""


class InputValidationError(OpenFEAPIError):
    """Raised when user input violates the input contract.

    Messages must name the offending file, chain or residue, state which rule was
    broken, and say how to fix it.
    """


class CampaignStateError(OpenFEAPIError):
    """Raised when a campaign directory is missing, malformed, or in the wrong state."""
