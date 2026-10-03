"""openfe-api: run OpenFE free energy experiments on HPC and server VMs."""

from importlib.metadata import PackageNotFoundError, version

from openfe_api.campaign import Campaign, RunState
from openfe_api.exceptions import CampaignStateError, InputValidationError, OpenFEAPIError
from openfe_api.schema.abfe import AbfeRequest
from openfe_api.schema.md import MdRequest
from openfe_api.schema.rbfe import RbfeRequest
from openfe_api.schema.request import CampaignRequest, load_request, validate_request
from openfe_api.schema.septop import SepTopRequest

try:
    __version__ = version("openfe-api")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"

__all__ = [
    "AbfeRequest",
    "Campaign",
    "CampaignRequest",
    "CampaignStateError",
    "InputValidationError",
    "MdRequest",
    "OpenFEAPIError",
    "RbfeRequest",
    "RunState",
    "SepTopRequest",
    "__version__",
    "load_request",
    "validate_request",
]
