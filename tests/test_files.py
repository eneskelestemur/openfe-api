"""Tests for reading a campaign's files back out."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from helpers import abfe_campaign
from openfe_api.campaign import SUBDIRECTORIES, Campaign
from openfe_api.exceptions import InputValidationError
from openfe_api.files import (
    DEFAULT_GROUPS,
    archive_size,
    list_files,
    resolve_file,
    stream_archive,
)


def _populate(campaign: Campaign) -> None:
    """Put one file in each group, including a deep path under runs/."""
    (campaign.directory / "results").mkdir(parents=True, exist_ok=True)
    (campaign.directory / "results" / "results.tsv").write_text("ligand\tdg\n", encoding="utf-8")
    (campaign.directory / "prepared").mkdir(parents=True, exist_ok=True)
    (campaign.directory / "prepared" / "protein.pdb").write_text("ATOM\n", encoding="utf-8")
    deep = campaign.directory / "runs" / "mini" / "repeat1" / ("shared_" + "u" * 80)
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "complex.nc").write_bytes(b"\x00" * 4096)


def test_a_group_is_a_campaign_subdirectory() -> None:
    """The client vocabulary is the layout, so there is no mapping to drift."""
    assert set(SUBDIRECTORIES) == {
        "inputs",
        "prepared",
        "plans",
        "relaxed",
        "runs",
        "results",
        "logs",
    }


def test_runs_is_the_only_group_left_out_by_default() -> None:
    assert "runs" not in DEFAULT_GROUPS
    assert set(DEFAULT_GROUPS) == set(SUBDIRECTORIES) - {"runs"}


def test_listing_reports_paths_relative_to_the_campaign(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)
    _populate(campaign)

    listed = list_files(campaign)

    paths = [entry.path for entry in listed]
    assert "results/results.tsv" in paths
    assert all(not Path(path).is_absolute() for path in paths)
    assert paths == sorted(paths)


def test_listing_can_be_narrowed_to_one_group(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)
    _populate(campaign)

    listed = list_files(campaign, ["results"])

    assert [entry.path for entry in listed] == ["results/results.tsv"]
    assert listed[0].group == "results"
    assert listed[0].size == len("ligand\tdg\n")


def test_listing_an_empty_group_is_not_an_error(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)

    assert list_files(campaign, ["relaxed"]) == []


def test_listing_refuses_an_unknown_group(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)

    with pytest.raises(InputValidationError, match="unknown file group"):
        list_files(campaign, ["trajectories"])


def test_resolving_returns_a_file_inside_the_campaign(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)
    _populate(campaign)

    resolved = resolve_file(campaign, "results/results.tsv")

    assert resolved.is_file()
    assert resolved.is_relative_to(campaign.directory.resolve())


@pytest.mark.parametrize(
    "path",
    ["../outside.txt", "/etc/passwd", "results/../../outside.txt", "", "results"],
)
def test_resolving_refuses_anything_outside_the_campaign(tmp_path: Path, path: str) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(InputValidationError):
        resolve_file(campaign, path)


def test_resolving_refuses_a_symlink_that_escapes(tmp_path: Path) -> None:
    """resolve() follows the link, so the escape is caught rather than served."""
    campaign = abfe_campaign(tmp_path, planned=False)
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    link = campaign.directory / "results" / "link.txt"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(tmp_path / "outside.txt")

    with pytest.raises(InputValidationError):
        resolve_file(campaign, "results/link.txt")


def test_a_symlink_is_left_out_of_the_listing(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)
    _populate(campaign)
    (campaign.directory / "results" / "link.tsv").symlink_to(
        campaign.directory / "results" / "results.tsv"
    )

    assert [entry.path for entry in list_files(campaign, ["results"])] == ["results/results.tsv"]


def test_the_archive_carries_the_listed_files(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)
    _populate(campaign)
    listed = list_files(campaign, ["results", "prepared"])

    blob = b"".join(stream_archive(campaign, listed))

    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        assert sorted(archive.getnames()) == sorted(entry.path for entry in listed)
        member = archive.extractfile("results/results.tsv")
        assert member is not None
        assert member.read() == b"ligand\tdg\n"


def test_the_predicted_size_is_the_exact_stream_length(tmp_path: Path) -> None:
    """A plain tar's length is promised as Content-Length, so it has to be right."""
    campaign = abfe_campaign(tmp_path, planned=False)
    _populate(campaign)
    listed = list_files(campaign)

    blob = b"".join(stream_archive(campaign, listed))

    assert archive_size(listed) == len(blob)


def test_the_predicted_size_holds_for_a_path_too_long_for_a_tar_header(tmp_path: Path) -> None:
    """The deep runs/ path needs a GNU long-name member, which also counts toward the size."""
    campaign = abfe_campaign(tmp_path, planned=False)
    _populate(campaign)
    listed = list_files(campaign, ["runs"])

    assert max(len(entry.path) for entry in listed) > 100
    assert archive_size(listed) == len(b"".join(stream_archive(campaign, listed)))


def test_the_archive_can_be_gzipped(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)
    _populate(campaign)
    listed = list_files(campaign, ["results"])

    blob = b"".join(stream_archive(campaign, listed, compress=True))

    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        assert archive.getnames() == ["results/results.tsv"]


@pytest.mark.parametrize("change", ["vanish", "grow", "shrink"])
def test_the_promised_length_holds_when_a_file_changes_mid_stream(
    tmp_path: Path, change: str
) -> None:
    """A job may still be writing, and Content-Length was already sent."""
    campaign = abfe_campaign(tmp_path, planned=False)
    _populate(campaign)
    listed = list_files(campaign, ["results", "prepared"])
    target = campaign.directory / "results" / "results.tsv"
    if change == "vanish":
        target.unlink()
    elif change == "grow":
        target.write_text("ligand\tdg\n" * 400, encoding="utf-8")
    else:
        target.write_text("x", encoding="utf-8")

    blob = b"".join(stream_archive(campaign, listed))

    assert archive_size(listed) == len(blob)
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        assert sorted(archive.getnames()) == sorted(entry.path for entry in listed)
