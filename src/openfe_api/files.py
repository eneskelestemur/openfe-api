"""Reading a campaign's files back out, for a client that cannot see its filesystem.

The archive is an uncompressed tar, so its length is known before the first byte goes out.
Simulation output is dense binary, where gzip costs far more time than it saves.
"""

from __future__ import annotations

import tarfile
from collections.abc import Buffer, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from io import RawIOBase
from pathlib import Path, PurePosixPath

from openfe_api.campaign import RUNS_DIR, SUBDIRECTORIES, Campaign
from openfe_api.exceptions import InputValidationError
from openfe_api.log import get_logger

__all__ = [
    "DEFAULT_GROUPS",
    "CampaignFile",
    "archive_size",
    "list_files",
    "resolve_file",
    "stream_archive",
]

logger = get_logger(__name__)

DEFAULT_GROUPS = tuple(group for group in SUBDIRECTORIES if group != RUNS_DIR)
"""Groups an archive carries unless the caller asks for others.

Simulation output is left out: a production campaign keeps gigabytes under ``runs``, and
those files are better pulled one at a time, where a dropped connection can resume.
"""

BLOCK = tarfile.BLOCKSIZE
RECORD = tarfile.RECORDSIZE


@dataclass(frozen=True)
class CampaignFile:
    """One file inside a campaign.

    Attributes:
        path: Location within the campaign, as a relative POSIX path.
        group: The campaign subdirectory it sits in.
        size: Size in bytes.
        modified: When it was last written.
    """

    path: str
    group: str
    size: int
    modified: datetime


def _blocks(size: int) -> int:
    """Round a byte count up to whole tar blocks."""
    return -(-size // BLOCK) * BLOCK


def _check_groups(groups: tuple[str, ...]) -> None:
    """Refuse a group that is not one of the campaign's subdirectories."""
    unknown = sorted(set(groups) - set(SUBDIRECTORIES))
    if unknown:
        raise InputValidationError(
            f"unknown file group(s) {unknown}; choose from {sorted(SUBDIRECTORIES)}"
        )


def list_files(
    campaign: Campaign, groups: list[str] | tuple[str, ...] | None = None
) -> list[CampaignFile]:
    """List the files a campaign holds.

    Args:
        campaign: The campaign to look in.
        groups: Subdirectories to include. All of them when None.

    Returns:
        One entry per file, sorted by path. A group whose directory does not exist
        contributes nothing.

    Raises:
        InputValidationError: If a group name is not recognized.
    """
    wanted = SUBDIRECTORIES if groups is None else tuple(groups)
    _check_groups(wanted)

    root = campaign.directory.resolve()
    found: list[CampaignFile] = []
    for group in wanted:
        directory = root / group
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            stat = path.stat()
            found.append(
                CampaignFile(
                    path=str(PurePosixPath(path.relative_to(root))),
                    group=group,
                    size=stat.st_size,
                    modified=datetime.fromtimestamp(stat.st_mtime, UTC),
                )
            )
    return sorted(found, key=lambda entry: entry.path)


def resolve_file(campaign: Campaign, relative: str) -> Path:
    """Turn a client-supplied relative path into a file inside the campaign.

    Args:
        campaign: The campaign the path is relative to.
        relative: Path as the client sent it.

    Returns:
        The absolute path of an existing regular file.

    Raises:
        InputValidationError: If the path leaves the campaign, names something that is not
            a regular file, or does not exist.
    """
    root = campaign.directory.resolve()
    candidate = (root / relative).resolve()

    # resolve() has already followed any symlink, so a link pointing outside the campaign
    # fails here along with the obvious '..' attempts.
    if not candidate.is_relative_to(root) or candidate == root:
        raise InputValidationError(f"'{relative}' is not a path inside the campaign")
    if not candidate.is_file():
        raise InputValidationError(f"'{relative}' is not a file in the campaign")
    return candidate


def archive_size(files: list[CampaignFile]) -> int:
    """Return the exact byte length of the tar stream these files produce.

    A client gets a real ``Content-Length``, so it can show progress and notice a truncated
    download.

    Args:
        files: The files the archive will carry.

    Returns:
        Length in bytes of the uncompressed tar.
    """
    total = 0
    for entry in files:
        # GNU tar writes a long name as an extra member before the real one.
        if len(entry.path.encode()) > tarfile.LENGTH_NAME:
            total += BLOCK + _blocks(len(entry.path.encode()) + 1)
        total += BLOCK + _blocks(entry.size)
    total += 2 * BLOCK
    return -(-total // RECORD) * RECORD


@contextmanager
def _tar_stream(buffer: _Buffer, compress: bool) -> Iterator[tarfile.TarFile]:
    """Open a streaming tar over a buffer, gzipped or not."""
    if compress:
        with tarfile.open(fileobj=buffer, mode="w|gz", format=tarfile.GNU_FORMAT) as archive:
            yield archive
    else:
        with tarfile.open(fileobj=buffer, mode="w|", format=tarfile.GNU_FORMAT) as archive:
            yield archive


def stream_archive(
    campaign: Campaign, files: list[CampaignFile], compress: bool = False
) -> Iterator[bytes]:
    """Produce a tar of the given files, a chunk at a time.

    Nothing is held in memory beyond one chunk, so the size of the campaign does not matter.

    Args:
        campaign: The campaign the files belong to.
        files: The files to carry, as returned by :func:`list_files`.
        compress: Whether to gzip the stream. Off by default: simulation output is already
            dense binary, so it costs minutes per gigabyte to save a few percent.

    Yields:
        Chunks of the tar stream.
    """
    buffer = _Buffer()
    root = campaign.directory.resolve()

    with _tar_stream(buffer, compress) as archive:
        for entry in files:
            info = tarfile.TarInfo(entry.path)
            info.size = entry.size
            info.mtime = int(entry.modified.timestamp())
            archive.addfile(info, _Exactly(root / entry.path, entry.size, entry.path))
            yield from buffer.drain()
    yield from buffer.drain()


class _Exactly(RawIOBase):
    """Reads one file, always yielding the number of bytes the listing recorded.

    A job may be writing while the archive streams, so a file can grow or disappear between
    the listing and the read. ``Content-Length`` is sent before that, so the member is
    truncated or padded to match rather than breaking the response.
    """

    def __init__(self, path: Path, size: int, label: str) -> None:
        super().__init__()
        self._handle = path.open("rb") if path.is_file() else None
        self._left = size
        self._label = label
        if self._handle is None and size:
            logger.warning("'%s' went away mid-archive; it is padded with zeros", label)

    def readable(self) -> bool:
        """Report that this file can be read."""
        return True

    def read(self, size: int = -1) -> bytes:
        """Return up to ``size`` bytes, padding once the file runs short."""
        if self._left <= 0:
            return b""
        wanted = self._left if size is None or size < 0 else min(size, self._left)
        data = self._handle.read(wanted) if self._handle is not None else b""
        if len(data) < wanted:
            if self._handle is not None:
                logger.warning("'%s' shrank mid-archive; it is padded with zeros", self._label)
            data += bytes(wanted - len(data))
        self._left -= len(data)
        if self._left <= 0 and self._handle is not None:
            self._handle.close()
            self._handle = None
        return data


class _Buffer(RawIOBase):
    """A write-only file for tarfile, handing each write straight on to the caller."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[bytes] = []

    def writable(self) -> bool:
        """Report that this file accepts writes."""
        return True

    def write(self, data: Buffer) -> int:
        """Collect one write from tarfile."""
        chunk = bytes(data)
        self._chunks.append(chunk)
        return len(chunk)

    def drain(self) -> Iterator[bytes]:
        """Yield everything collected since the last drain."""
        chunks, self._chunks = self._chunks, []
        if chunks:
            yield b"".join(chunks)
