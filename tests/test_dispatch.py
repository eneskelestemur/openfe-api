"""Tests that every stage refuses a protocol it does not handle.

Each stage used to treat anything that was not ABFE as RBFE. Half of a SepTop campaign would
have been handled correctly by that branch and half silently wrongly, so the invariant worth
holding is that an unhandled protocol fails loudly at every seam.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from openfe_api.campaign import Campaign
from openfe_api.exceptions import OpenFEAPIError
from openfe_api.execution.base import build_tasks
from openfe_api.prep.runner import prepare_campaign
from openfe_api.protocols.runner import plan_campaign
from openfe_api.results.runner import gather_campaign
from openfe_api.schema.common import RelaxSpec
from openfe_api.schema.request import validate_request

DATA = Path(__file__).parent / "data"

STAGES = {
    "prep": lambda campaign: prepare_campaign(campaign, check_parameters=False),
    "plan": plan_campaign,
    "execute": build_tasks,
    "gather": gather_campaign,
}


@pytest.fixture
def campaign_of_an_unknown_protocol(
    series_request_data: dict[str, Any], tmp_path: Path
) -> Campaign:
    """Create a campaign whose manifest names a protocol no stage implements.

    Args:
        series_request_data: Any valid request, used to build the directory.
        tmp_path: Pytest temporary directory.

    Returns:
        The campaign, with its in-memory request replaced by a stub.
    """
    campaign = Campaign.create(tmp_path / "campaign", validate_request(series_request_data))
    campaign.manifest.protocol = "plainmd"
    campaign.manifest.request = cast(
        Any, SimpleNamespace(protocol="plainmd", relax=RelaxSpec(), execution=None)
    )
    return campaign


@pytest.mark.parametrize("stage", sorted(STAGES), ids=sorted(STAGES))
def test_a_stage_refuses_a_protocol_it_does_not_handle(
    stage: str, campaign_of_an_unknown_protocol: Campaign
) -> None:
    with pytest.raises(OpenFEAPIError, match="plainmd"):
        STAGES[stage](campaign_of_an_unknown_protocol)


@pytest.mark.parametrize("protocol", ["abfe", "rbfe", "septop", "md"])
def test_the_union_accepts_every_implemented_protocol(
    protocol: str,
    combined_request_data: dict[str, Any],
    series_request_data: dict[str, Any],
    md_request_data: dict[str, Any],
) -> None:
    by_protocol = {
        "abfe": combined_request_data,
        "rbfe": dict(series_request_data),
        "septop": dict(series_request_data),
        "md": md_request_data,
    }
    data = by_protocol[protocol]
    data["protocol"] = protocol

    assert validate_request(data).protocol == protocol


def test_an_unknown_protocol_names_the_implemented_ones(
    series_request_data: dict[str, Any],
) -> None:
    series_request_data["protocol"] = "plainmd"

    with pytest.raises(OpenFEAPIError) as caught:
        validate_request(series_request_data)

    message = str(caught.value)
    for protocol in ("'abfe'", "'rbfe'", "'septop'", "'md'"):
        assert protocol in message
