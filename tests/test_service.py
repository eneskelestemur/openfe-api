"""Tests for the HTTP service."""

from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from helpers import write_md_result, write_phase_result, write_result
from openfe_api.campaign import Campaign, RunState
from openfe_api.config import ServiceSettings
from openfe_api.schema.abfe import AbfeRequest
from openfe_api.schema.md import MdRequest
from openfe_api.schema.request import validate_request
from openfe_api.service import create_app

DATA = Path(__file__).parent / "data"


def _request_body(name: str = "svc_campaign") -> dict[str, Any]:
    """Build a campaign request body pointing at the mini fixture.

    Args:
        name: Campaign name.

    Returns:
        The request as a JSON-ready dictionary.
    """
    return {
        "protocol": "abfe",
        "name": name,
        "complexes": [
            {
                "name": "mini",
                "structure": str(DATA / "mini_complex.cif"),
                "protein": {"chains": ["A"]},
                "ligand": {"selector": {"chain": "L"}, "smiles": "C[C@H](O)c1ccccc1"},
                "extra_ligand_copies": "drop",
            }
        ],
        "execution": {"repeats": 1},
    }


def _series_body(name: str = "svc_series") -> dict[str, Any]:
    """Build an RBFE campaign request body pointing at the mini fixture.

    Args:
        name: Campaign name.

    Returns:
        The request as a JSON-ready dictionary.
    """
    return {
        "protocol": "rbfe",
        "name": name,
        "protein": {"chains": ["A"]},
        "ligands": [
            {
                "name": "ref",
                "structure": str(DATA / "mini_complex.cif"),
                "selector": {"chain": "L"},
                "smiles": "C[C@H](O)c1ccccc1",
            },
            {"name": "fluoro", "smiles": "C[C@H](O)c1ccc(F)cc1"},
        ],
        "execution": {"repeats": 1},
    }


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """Build a test client backed by a temporary campaigns directory.

    Args:
        tmp_path: Pytest temporary directory.

    Yields:
        The client.
    """
    settings = ServiceSettings(root=tmp_path / "campaigns", check_parameters=False)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_health_reports_the_version(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"]


def test_listing_is_empty_at_first(client: TestClient) -> None:
    assert client.get("/campaigns").json() == []


def test_create_campaign(client: TestClient) -> None:
    response = client.post("/campaigns", json=_request_body())

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "svc_campaign"
    assert body["runs"][0]["state"] == "pending"
    assert client.get("/campaigns").json() == ["svc_campaign"]


def test_create_rejects_an_invalid_request(client: TestClient) -> None:
    body = _request_body()
    del body["complexes"][0]["extra_ligand_copies"]

    response = client.post("/campaigns", json=body)

    assert response.status_code == 422
    assert "extra_ligand_copies" in response.text


def test_create_reports_missing_input_files(client: TestClient) -> None:
    body = _request_body()
    body["complexes"][0]["structure"] = "/nowhere/model.cif"

    response = client.post("/campaigns", json=body)

    assert response.status_code == 422
    assert "input files not found" in response.text


def test_create_refuses_a_duplicate(client: TestClient) -> None:
    client.post("/campaigns", json=_request_body())

    response = client.post("/campaigns", json=_request_body())

    assert response.status_code == 409


def test_create_can_overwrite_with_force(client: TestClient) -> None:
    client.post("/campaigns", json=_request_body())

    response = client.post("/campaigns?force=true", json=_request_body())

    assert response.status_code == 201


def test_get_campaign_reports_runs(client: TestClient) -> None:
    client.post("/campaigns", json=_request_body())

    response = client.get("/campaigns/svc_campaign")

    assert response.status_code == 200
    assert response.json()["runs"][0]["name"] == "mini"


def test_get_unknown_campaign_is_404(client: TestClient) -> None:
    assert client.get("/campaigns/absent").status_code == 404


@pytest.mark.parametrize(
    "name",
    ["..%2F..%2Fetc", "..%2Fsibling", ".hidden", "with%20space", "%2Fabsolute"],
)
def test_a_name_that_is_not_a_campaign_name_is_refused(client: TestClient, name: str) -> None:
    """The name is joined onto the campaigns root, so only the name pattern is accepted."""
    assert client.get(f"/campaigns/{name}").status_code in (404, 422)


def test_a_campaign_outside_the_root_stays_unreachable(client: TestClient, tmp_path: Path) -> None:
    """End to end: a real campaign one level above the root cannot be opened through the path."""
    outside = tmp_path / "outside"
    Campaign.create(outside, validate_request(_request_body("outside")))

    response = client.get("/campaigns/..%2Foutside")

    assert response.status_code == 404


def test_prep_runs_in_the_background(client: TestClient, tmp_path: Path) -> None:
    client.post("/campaigns", json=_request_body())

    response = client.post("/campaigns/svc_campaign/prep")

    assert response.status_code == 202
    assert response.json()["operation"] == "prep"

    summary = client.get("/campaigns/svc_campaign").json()
    assert summary["last_operation"]["status"] == "done"
    assert summary["runs"][0]["state"] == "prepared"
    assert (tmp_path / "campaigns" / "svc_campaign" / "prepared" / "mini" / "ligand.sdf").is_file()


def test_prep_records_a_failure(client: TestClient) -> None:
    body = _request_body()
    body["complexes"][0]["ligand"]["smiles"] = "CCO"
    client.post("/campaigns", json=body)

    client.post("/campaigns/svc_campaign/prep")

    summary = client.get("/campaigns/svc_campaign").json()
    assert summary["runs"][0]["state"] == "failed"
    assert "heavy atoms" in summary["runs"][0]["message"]


def test_plan_before_prep_is_recorded_as_a_failure(client: TestClient) -> None:
    client.post("/campaigns", json=_request_body())

    client.post("/campaigns/svc_campaign/plan")

    summary = client.get("/campaigns/svc_campaign").json()
    assert summary["last_operation"]["operation"] == "plan"
    assert "failed 1" in summary["last_operation"]["message"]


def test_submit_without_a_plan_fails_visibly(client: TestClient) -> None:
    client.post("/campaigns", json=_request_body())

    response = client.post("/campaigns/svc_campaign/submit")

    assert response.status_code == 202
    summary = client.get("/campaigns/svc_campaign").json()
    assert summary["last_operation"]["status"] == "failed"
    assert "no planned transformations" in summary["last_operation"]["message"]


def test_submit_reports_an_unknown_profile(client: TestClient) -> None:
    body = _request_body()
    body["execution"] = {"profile": "absent", "repeats": 1}
    client.post("/campaigns", json=body)

    response = client.post("/campaigns/svc_campaign/submit")

    assert response.status_code == 422
    assert "unknown execution profile" in response.text


def test_results_are_empty_before_anything_runs(client: TestClient) -> None:
    client.post("/campaigns", json=_request_body())

    assert client.get("/campaigns/svc_campaign/results").json() == []


def test_results_report_free_energies_and_quality(client: TestClient, tmp_path: Path) -> None:
    client.post("/campaigns", json=_request_body())
    campaign = Campaign.open(tmp_path / "campaigns" / "svc_campaign")
    campaign.set_state("mini", RunState.PREPARED)
    campaign.set_state("mini", RunState.PLANNED)
    write_result(campaign.directory / "runs" / "mini" / "repeat1" / "results.json", estimate=-7.5)

    body = client.get("/campaigns/svc_campaign/results").json()

    assert len(body) == 1
    assert body[0]["dg"] == pytest.approx(-7.5)
    assert body[0]["quality"] in {"pass", "unknown", "fail"}
    assert any(check["name"] == "mbar_overlap" for check in body[0]["checks"])


def test_openapi_schema_is_served(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert schema["info"]["title"] == "openfe-api"
    assert "/campaigns/{name}/prep" in schema["paths"]


def _plan_series(campaign: Campaign, edges: list[tuple[str, str]]) -> None:
    """Write the network summary and edge runs planning would have created.

    Args:
        campaign: The campaign to plan.
        edges: Ligand name pairs.
    """
    plans = campaign.directory / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    summary = {
        "edges": [{"name": f"{a}_to_{b}", "ligand_a": a, "ligand_b": b} for a, b in edges],
        "dropped": {},
        "unreachable": [],
        "warnings": [],
    }
    (plans / "network.json").write_text(json.dumps(summary), encoding="utf-8")
    campaign.add_runs([f"{a}_to_{b}" for a, b in edges], repeats=1, state=RunState.PREPARED)
    for a, b in edges:
        campaign.set_state(f"{a}_to_{b}", RunState.PLANNED)


def test_an_rbfe_campaign_can_be_created(client: TestClient) -> None:
    response = client.post("/campaigns", json=_series_body())

    assert response.status_code == 201
    assert response.json()["protocol"] == "rbfe"


def test_abfe_results_are_refused_for_a_series(client: TestClient) -> None:
    """Relative values must not be served under a field meaning absolute ones."""
    client.post("/campaigns", json=_series_body())

    response = client.get("/campaigns/svc_series/results")

    assert response.status_code == 409
    assert "/network" in response.json()["detail"]


def test_the_network_endpoint_is_refused_for_a_complex_campaign(client: TestClient) -> None:
    client.post("/campaigns", json=_request_body())

    response = client.get("/campaigns/svc_campaign/network")

    assert response.status_code == 409
    assert "/results" in response.json()["detail"]


def test_the_network_endpoint_needs_a_plan(client: TestClient) -> None:
    client.post("/campaigns", json=_series_body())

    response = client.get("/campaigns/svc_series/network")

    assert response.status_code == 409
    assert "no planned network" in response.json()["detail"]


def test_the_network_endpoint_reports_edges_and_cycles(client: TestClient, tmp_path: Path) -> None:
    client.post("/campaigns", json=_series_body())
    campaign = Campaign.open(tmp_path / "campaigns" / "svc_series")
    _plan_series(campaign, [("a", "b"), ("b", "c"), ("a", "c")])
    for edge, estimate in (("a_to_b", 1.0), ("b_to_c", 2.0), ("a_to_c", 3.5)):
        for phase, value in (("solvent", 0.0), ("complex", estimate)):
            write_phase_result(
                campaign.directory / "runs" / edge / phase / "repeat1" / "results.json",
                value,
            )

    body = client.get("/campaigns/svc_series/network").json()

    assert len(body["edges"]) == 3
    assert {entry["ligand"] for entry in body["ligands"]} == {"a", "b", "c"}
    assert len(body["cycles"]) == 1
    assert abs(body["cycles"][0]["closure"]) == pytest.approx(0.5, abs=1e-6)
    assert body["unreachable"] == []
    assert any(check["name"] == "cycle_closure" for check in body["checks"])


def test_the_network_endpoint_names_unreachable_ligands(client: TestClient, tmp_path: Path) -> None:
    client.post("/campaigns", json=_series_body())
    campaign = Campaign.open(tmp_path / "campaigns" / "svc_series")
    _plan_series(campaign, [("a", "b"), ("c", "d")])
    for phase, value in (("solvent", 0.0), ("complex", 1.0)):
        write_phase_result(
            campaign.directory / "runs" / "a_to_b" / phase / "repeat1" / "results.json",
            value,
        )

    body = client.get("/campaigns/svc_series/network").json()

    assert set(body["unreachable"]) == {"c", "d"}
    assert any(entry["note"] for entry in body["edges"] if entry["edge"] == "c_to_d")


def _upload_body(name: str = "svc_campaign") -> dict[str, Any]:
    """An ABFE request naming its structure by bare filename, as an upload must."""
    body = _request_body(name)
    body["complexes"][0]["structure"] = "model.cif"
    return body


def _upload(
    client: TestClient,
    body: dict[str, Any] | None = None,
    files: list[tuple[str, tuple[str, bytes]]] | None = None,
) -> Any:
    """Post one upload, defaulting to a valid request and the mini fixture."""
    return client.post(
        "/campaigns/upload",
        data={"campaign": json.dumps(body if body is not None else _upload_body())},
        files=files
        if files is not None
        else [("files", ("model.cif", (DATA / "mini_complex.cif").read_bytes()))],
    )


def test_upload_creates_a_campaign_that_owns_its_inputs(client: TestClient, tmp_path: Path) -> None:
    response = _upload(client)

    assert response.status_code == 201, response.text
    campaign = Campaign.open(tmp_path / "campaigns" / "svc_campaign")
    spec = cast(AbfeRequest, campaign.manifest.request).complexes[0]
    assert spec.structure is not None
    assert spec.structure == campaign.directory / "inputs" / "model.cif"
    assert spec.structure.read_bytes() == (DATA / "mini_complex.cif").read_bytes()
    assert spec.ligand.smiles == "C[C@H](O)c1ccccc1"


def test_upload_repoints_every_protocol_not_only_abfe(client: TestClient, tmp_path: Path) -> None:
    """The paths are rewritten through ``path_holders``, so a plain MD request works too."""
    body = {
        "protocol": "md",
        "name": "svc_md",
        "systems": [
            {
                "name": "holo",
                "structure": "model.cif",
                "protein": {"chains": ["A"]},
                "ligands": [
                    {"name": "lig", "selector": {"chain": "L"}, "smiles": "C[C@H](O)c1ccccc1"}
                ],
            }
        ],
        "execution": {"repeats": 1},
    }

    response = _upload(client, body)

    assert response.status_code == 201, response.text
    system = cast(MdRequest, Campaign.open(tmp_path / "campaigns" / "svc_md").manifest.request)
    assert system.systems[0].structure is not None
    assert system.systems[0].structure.is_file()


def test_upload_accepts_split_files_and_cofactors(client: TestClient, tmp_path: Path) -> None:
    body = _request_body()
    body["complexes"] = [
        {
            "name": "split",
            "protein": {"path": "protein.pdb"},
            "ligand": {"path": "ligand.sdf", "smiles": "CCO"},
            "cofactors": [{"name": "cofactor", "path": "cofactor.sdf", "smiles": "CC"}],
        }
    ]

    response = _upload(
        client,
        body,
        [
            ("files", (name, b"coordinates"))
            for name in ("protein.pdb", "ligand.sdf", "cofactor.sdf")
        ],
    )

    assert response.status_code == 201, response.text
    request = cast(
        AbfeRequest, Campaign.open(tmp_path / "campaigns" / "svc_campaign").manifest.request
    )
    assert request.complexes[0].protein.path == (
        tmp_path / "campaigns" / "svc_campaign" / "inputs" / "protein.pdb"
    )


def test_upload_never_overwrites_an_existing_campaign(client: TestClient) -> None:
    assert _upload(client).status_code == 201

    assert _upload(client).status_code == 409


@pytest.mark.parametrize("filename", ["../model.cif", "/tmp/model.cif", "model.exe", ".model.cif"])
def test_upload_refuses_a_filename_that_is_not_a_bare_coordinate_file(
    client: TestClient, filename: str
) -> None:
    assert _upload(client, files=[("files", (filename, b"coordinates"))]).status_code == 422


@pytest.mark.parametrize("path", ["/tmp/model.cif", "../model.cif", "inputs/model.cif"])
def test_upload_refuses_a_request_that_points_at_the_server(client: TestClient, path: str) -> None:
    body = _upload_body()
    body["complexes"][0]["structure"] = path

    assert _upload(client, body).status_code == 422


def test_upload_refuses_files_that_do_not_match_the_request(client: TestClient) -> None:
    assert _upload(client, files=[("files", ("other.cif", b"coordinates"))]).status_code == 422


def test_upload_refuses_duplicate_filenames(client: TestClient) -> None:
    duplicated = [("files", ("model.cif", b"coordinates"))] * 2

    assert _upload(client, files=duplicated).status_code == 422


def test_upload_refuses_an_empty_file_and_leaves_nothing_behind(
    client: TestClient, tmp_path: Path
) -> None:
    assert _upload(client, files=[("files", ("model.cif", b""))]).status_code == 422

    assert not (tmp_path / "campaigns" / "svc_campaign").exists()


def test_upload_enforces_the_size_limit_and_leaves_nothing_behind(tmp_path: Path) -> None:
    settings = ServiceSettings(
        root=tmp_path / "campaigns", check_parameters=False, max_upload_bytes=4
    )

    with TestClient(create_app(settings)) as small:
        assert _upload(small).status_code == 413

    assert not (tmp_path / "campaigns" / "svc_campaign").exists()


def test_upload_refuses_a_campaign_field_that_is_not_json(client: TestClient) -> None:
    response = client.post(
        "/campaigns/upload",
        data={"campaign": "not json"},
        files=[("files", ("model.cif", b"coordinates"))],
    )

    assert response.status_code == 422


def _with_files(client: TestClient, tmp_path: Path) -> Path:
    """Create a campaign through the service and leave a file in two of its groups."""
    client.post("/campaigns", json=_request_body())
    directory = tmp_path / "campaigns" / "svc_campaign"
    (directory / "results").mkdir(parents=True, exist_ok=True)
    (directory / "results" / "results.tsv").write_text("ligand\tdg\n", encoding="utf-8")
    (directory / "runs" / "mini" / "repeat1").mkdir(parents=True, exist_ok=True)
    (directory / "runs" / "mini" / "repeat1" / "complex.nc").write_bytes(b"\x00" * 2048)
    return directory


def test_files_are_listed_with_sizes(client: TestClient, tmp_path: Path) -> None:
    _with_files(client, tmp_path)

    body = client.get("/campaigns/svc_campaign/files").json()

    entry = next(item for item in body if item["path"] == "results/results.tsv")
    assert entry["group"] == "results"
    assert entry["size"] == len("ligand\tdg\n")
    assert any(item["group"] == "runs" for item in body)


def test_files_can_be_listed_by_group(client: TestClient, tmp_path: Path) -> None:
    _with_files(client, tmp_path)

    body = client.get("/campaigns/svc_campaign/files", params={"group": "results"}).json()

    assert [item["path"] for item in body] == ["results/results.tsv"]


def test_listing_an_unknown_group_is_refused(client: TestClient, tmp_path: Path) -> None:
    _with_files(client, tmp_path)

    response = client.get("/campaigns/svc_campaign/files", params={"group": "nope"})

    assert response.status_code == 422
    assert "unknown file group" in response.text


def test_a_file_can_be_downloaded(client: TestClient, tmp_path: Path) -> None:
    _with_files(client, tmp_path)

    response = client.get("/campaigns/svc_campaign/files/results/results.tsv")

    assert response.status_code == 200
    assert response.content == b"ligand\tdg\n"


@pytest.mark.parametrize("path", ["../../outside.txt", "results/../../outside.txt"])
def test_downloading_outside_the_campaign_is_refused(
    client: TestClient, tmp_path: Path, path: str
) -> None:
    _with_files(client, tmp_path)
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")

    response = client.get(f"/campaigns/svc_campaign/files/{path}")

    assert response.status_code in (404, 422)
    assert b"secret" not in response.content


def test_downloading_a_missing_file_is_refused(client: TestClient, tmp_path: Path) -> None:
    _with_files(client, tmp_path)

    assert client.get("/campaigns/svc_campaign/files/results/absent.tsv").status_code == 422


def test_the_archive_leaves_simulation_output_out_by_default(
    client: TestClient, tmp_path: Path
) -> None:
    _with_files(client, tmp_path)

    response = client.get("/campaigns/svc_campaign/archive")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-tar"
    with tarfile.open(fileobj=io.BytesIO(response.content)) as archive:
        names = archive.getnames()
    assert "results/results.tsv" in names
    assert not any(name.startswith("runs/") for name in names)


def test_the_archive_states_its_length_up_front(client: TestClient, tmp_path: Path) -> None:
    _with_files(client, tmp_path)

    response = client.get("/campaigns/svc_campaign/archive")

    assert int(response.headers["content-length"]) == len(response.content)


def test_simulation_output_is_included_on_request(client: TestClient, tmp_path: Path) -> None:
    _with_files(client, tmp_path)

    response = client.get("/campaigns/svc_campaign/archive", params={"group": "runs"})

    with tarfile.open(fileobj=io.BytesIO(response.content)) as archive:
        assert archive.getnames() == ["runs/mini/repeat1/complex.nc"]


def test_the_archive_can_be_gzipped_on_request(client: TestClient, tmp_path: Path) -> None:
    _with_files(client, tmp_path)

    response = client.get("/campaigns/svc_campaign/archive", params={"compress": "true"})

    assert response.headers["content-type"] == "application/gzip"
    with tarfile.open(fileobj=io.BytesIO(response.content)) as archive:
        assert "results/results.tsv" in archive.getnames()


def test_the_archive_refuses_an_unknown_group(client: TestClient, tmp_path: Path) -> None:
    _with_files(client, tmp_path)

    assert (
        client.get("/campaigns/svc_campaign/archive", params={"group": "nope"}).status_code == 422
    )


def test_a_failed_upload_keeps_a_directory_it_did_not_create(
    client: TestClient, tmp_path: Path
) -> None:
    """Cleanup removes only what the upload made, so an existing directory survives."""
    directory = tmp_path / "campaigns" / "svc_campaign"
    directory.mkdir(parents=True)
    (directory / "keep.txt").write_text("mine", encoding="utf-8")

    assert _upload(client, files=[("files", ("model.cif", b""))]).status_code == 422

    assert (directory / "keep.txt").read_text(encoding="utf-8") == "mine"
    assert not (directory / "inputs").exists()


def _md_body(name: str = "svc_md") -> dict[str, Any]:
    """A plain MD request pointing at the mini fixture."""
    return {
        "protocol": "md",
        "name": name,
        "systems": [
            {
                "name": "holo",
                "structure": str(DATA / "mini_complex.cif"),
                "protein": {"chains": ["A"]},
                "ligands": [
                    {"name": "lig", "selector": {"chain": "L"}, "smiles": "C[C@H](O)c1ccccc1"}
                ],
            }
        ],
        "execution": {"repeats": 1},
    }


def test_simulation_artifacts_are_paths_a_client_can_fetch(
    client: TestClient, tmp_path: Path
) -> None:
    """They name files inside the campaign, so the download endpoint accepts them as they are."""
    client.post("/campaigns", json=_md_body())
    campaign = Campaign.open(tmp_path / "campaigns" / "svc_md")
    campaign.set_state("holo", RunState.PREPARED)
    campaign.set_state("holo", RunState.PLANNED)
    write_md_result(campaign.directory / "runs" / "holo" / "repeat1" / "results.json")

    body = client.get("/campaigns/svc_md/simulations").json()

    trajectory = body[0]["artifacts"]["trajectory"]
    assert not Path(trajectory).is_absolute()
    assert trajectory.startswith("runs/holo/repeat1/")
    (campaign.directory / trajectory).parent.mkdir(parents=True, exist_ok=True)
    (campaign.directory / trajectory).write_bytes(b"frames")
    assert client.get(f"/campaigns/svc_md/files/{trajectory}").content == b"frames"
