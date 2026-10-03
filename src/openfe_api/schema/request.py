"""The input contract: a validated description of one campaign.

A request names input files explicitly, never a whole prediction folder, and a SMILES is
required for every ligand and cofactor because structure-model outputs carry no bond orders and
their embedded SMILES cannot be trusted.

``protocol`` selects the shape of the rest of the request, so a field belonging to another
protocol is rejected rather than ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, TypeAdapter, ValidationError

from openfe_api.exceptions import InputValidationError
from openfe_api.schema.abfe import AbfeRequest
from openfe_api.schema.common import read_request_mapping
from openfe_api.schema.md import MdRequest
from openfe_api.schema.rbfe import RbfeRequest
from openfe_api.schema.septop import SepTopRequest

__all__ = ["CampaignRequest", "load_request", "validate_request"]

type CampaignRequest = Annotated[
    AbfeRequest | RbfeRequest | SepTopRequest | MdRequest, Field(discriminator="protocol")
]
"""Any validated campaign request, discriminated on ``protocol``."""

_ADAPTER: TypeAdapter[CampaignRequest] = TypeAdapter(CampaignRequest)

_PROTOCOLS = ("abfe", "rbfe", "septop", "md")


def _describe(error: ValidationError) -> str:
    """Render a pydantic validation error as one readable line per problem.

    Pydantic names a discriminated union by its full internal type, which tells a user nothing,
    so the union tag is dropped and each problem is reported as its field and the broken rule.
    """
    lines: list[str] = []
    for item in error.errors():
        location = [str(part) for part in item["loc"]]
        if location and location[0] in _PROTOCOLS:
            location = location[1:]
        where = ".".join(location) or "request"
        lines.append(f"{where}: {item['msg'].removeprefix('Value error, ')}")
    unique = list(dict.fromkeys(lines))
    return "invalid request:\n  " + "\n  ".join(unique)


def validate_request(data: Any) -> CampaignRequest:
    """Validate a request from already-parsed data.

    Args:
        data: A mapping describing the campaign, including ``protocol``.

    Returns:
        The validated request for the protocol named by ``protocol``.

    Raises:
        InputValidationError: If the data does not satisfy the contract.
    """
    try:
        return _ADAPTER.validate_python(data)
    except ValidationError as error:
        raise InputValidationError(_describe(error)) from error


def load_request(path: Path) -> CampaignRequest:
    """Load and validate a request from a YAML file.

    Relative input paths are resolved against the request file's directory, and every
    referenced file must exist.

    Args:
        path: Path to the request YAML file.

    Returns:
        The validated request, with absolute input paths.

    Raises:
        InputValidationError: If the file is missing, is not a YAML mapping, fails schema
            validation, or refers to a file that does not exist.
    """
    raw = read_request_mapping(path)
    try:
        request = _ADAPTER.validate_python(raw)
    except ValidationError as error:
        raise InputValidationError(f"{path}: {_describe(error)}") from error

    request.resolve_paths(path.parent)
    return request
