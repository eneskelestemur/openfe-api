"""Collecting what a plain MD run produced.

The protocol reports no free energy: ``get_estimate`` returns None by design. So gathering an
MD campaign means saying whether each repeat finished and naming the files it wrote. Metrics
over those trajectories belong to the analysis stage, not here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openfe_api.log import get_logger
from openfe_api.results.gather import TRAJECTORY_OUTPUT, load_result_json

__all__ = [
    "MdRepeatResult",
    "MdRunResult",
    "gather_md",
    "read_md_repeat",
    "write_md_tsv",
]

logger = get_logger(__name__)

ARTIFACT_KEYS = {
    "topology": "system_pdb",
    "minimized": "minimized_pdb",
    "nvt_structure": "nvt_equil_pdb",
    "npt_structure": "npt_equil_pdb",
    "trajectory": TRAJECTORY_OUTPUT,
    "checkpoint": "last_checkpoint",
}
"""What each artifact is called here, mapped to the output key the simulation unit uses."""


@dataclass
class MdRepeatResult:
    """One repeat of one system.

    Attributes:
        path: The result file this came from.
        repeat: One-based repeat index taken from the directory name.
        artifacts: Files the run produced, keyed by the names in :data:`ARTIFACT_KEYS`.
            An entry is absent when the run did not write it.
        failure: Why this repeat is unusable, if it is.
    """

    path: Path
    repeat: int
    artifacts: dict[str, Path] = field(default_factory=dict)
    failure: str | None = None

    @property
    def ok(self) -> bool:
        """Whether this repeat finished.

        Returns:
            True if there is no recorded failure and a trajectory was written.
        """
        return self.failure is None and "trajectory" in self.artifacts


@dataclass
class MdRunResult:
    """Everything gathered for one system.

    Attributes:
        name: System name.
        repeats: One entry per repeat that wrote a result file.
    """

    name: str
    repeats: list[MdRepeatResult] = field(default_factory=list)

    @property
    def finished(self) -> list[MdRepeatResult]:
        """Repeats that finished.

        Returns:
            The usable repeats, in repeat order.
        """
        return [entry for entry in self.repeats if entry.ok]

    @property
    def ok(self) -> bool:
        """Whether every repeat that reported finished.

        Returns:
            True if there is at least one repeat and none of them failed.
        """
        return bool(self.repeats) and len(self.finished) == len(self.repeats)


def _artifacts(outputs: dict[str, Any]) -> dict[str, Path]:
    """Pick the artifact paths out of one unit's outputs."""
    found: dict[str, Path] = {}
    for label, key in ARTIFACT_KEYS.items():
        value = outputs.get(key)
        if value:
            found[label] = Path(str(value))
    return found


def read_md_repeat(path: Path, repeat: int) -> MdRepeatResult:
    """Read one MD result file.

    A file recording a failed simulation is returned with ``failure`` set rather than
    raising, so one broken repeat does not hide the others.

    Args:
        path: The result JSON.
        repeat: One-based repeat index.

    Returns:
        The repeat result.

    Raises:
        OpenFEAPIError: If the file cannot be read at all.
    """
    result = load_result_json(path)
    outcome = MdRepeatResult(path=path, repeat=repeat)

    unit_results = result.get("unit_results", {})
    raised = [entry for entry in unit_results.values() if "exception" in entry]
    for entry in unit_results.values():
        outcome.artifacts.update(_artifacts(entry.get("outputs", {})))

    if "trajectory" not in outcome.artifacts:
        outcome.failure = (
            f"{len(raised)} protocol unit(s) raised an exception"
            if raised
            else "no trajectory was written; the simulation did not finish"
        )
    return outcome


def gather_md(campaign_dir: Path, names: list[str]) -> list[MdRunResult]:
    """Collect every repeat of every system of a plain MD campaign.

    Args:
        campaign_dir: The campaign directory.
        names: System names to collect, in the order they should appear.

    Returns:
        One entry per system, including systems with nothing to collect yet.
    """
    results: list[MdRunResult] = []
    for name in names:
        run = MdRunResult(name=name)
        for repeat_dir in sorted((campaign_dir / "runs" / name).glob("repeat*")):
            path = repeat_dir / "results.json"
            if not path.is_file():
                continue
            repeat = int(repeat_dir.name.removeprefix("repeat") or 0)
            run.repeats.append(read_md_repeat(path, repeat))
        results.append(run)

    finished = sum(len(run.finished) for run in results)
    logger.info("Gathered %d finished repeat(s) across %d system(s)", finished, len(results))
    return results


def write_md_tsv(results: list[MdRunResult], path: Path) -> None:
    """Write the artifacts of every repeat as a TSV file.

    Args:
        results: The systems to write, in the order they should appear.
        path: File to write to. Parent directories are created as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["system", "repeat", "status", *ARTIFACT_KEYS, "note"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        for run in results:
            for entry in run.repeats:
                writer.writerow(
                    [
                        run.name,
                        entry.repeat,
                        "finished" if entry.ok else "failed",
                        *(str(entry.artifacts.get(label, "")) for label in ARTIFACT_KEYS),
                        entry.failure or "",
                    ]
                )
    logger.info("Wrote %s", path)
