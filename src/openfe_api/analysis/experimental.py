"""Reading the measured free energies a campaign is compared against.

Values are read in kcal/mol and used as given. Converting an IC50 or a Kd into a free
energy rests on assumptions this project does not make for you, so the table has to arrive
already in that form.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from openfe_api.exceptions import InputValidationError
from openfe_api.log import get_logger

__all__ = ["Measurement", "read_measurements"]

logger = get_logger(__name__)

NAME_COLUMNS = ("name", "ligand")
VALUE_COLUMNS = ("dg", "dg (kcal/mol)", "experimental", "experimental_dg")
UNCERTAINTY_COLUMNS = ("uncertainty", "error", "dg_uncertainty", "uncertainty (kcal/mol)")


@dataclass(frozen=True)
class Measurement:
    """One ligand's measured binding free energy.

    Attributes:
        name: Ligand name, matching the campaign's.
        dg: Measured free energy in kcal/mol.
        uncertainty: Measurement error in kcal/mol, when the table carries one.
    """

    name: str
    dg: float
    uncertainty: float | None = None


def _column(header: list[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first header matching any candidate, ignoring case and spacing."""
    normalized = {name.strip().lower(): name for name in header}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _number(raw: str, name: str, column: str, path: Path) -> float:
    """Parse one cell as a free energy, naming the row that is wrong when it is not."""
    try:
        return float(raw)
    except ValueError as error:
        raise InputValidationError(
            f"{path}: '{column}' for ligand '{name}' is {raw!r}, which is not a number. "
            "Measured values are read in kcal/mol."
        ) from error


def read_measurements(path: Path) -> dict[str, Measurement]:
    """Read a table of measured free energies.

    The table is a CSV or TSV with a ligand name column (``name`` or ``ligand``), a value
    column in kcal/mol (``dg``, ``dg (kcal/mol)``, ``experimental`` or ``experimental_dg``)
    and optionally an uncertainty column. Column names are matched ignoring case.

    Args:
        path: The table to read.

    Returns:
        One measurement per ligand, keyed by name.

    Raises:
        InputValidationError: If the file is missing, has no usable columns, names a ligand
            twice, carries a value that is not a number, or holds no rows.
    """
    if not path.is_file():
        raise InputValidationError(f"no measured values at {path}")

    text = path.read_text(encoding="utf-8")
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    header = list(reader.fieldnames or [])

    name_column = _column(header, NAME_COLUMNS)
    value_column = _column(header, VALUE_COLUMNS)
    if name_column is None or value_column is None:
        raise InputValidationError(
            f"{path}: a measured-value table needs a ligand name column "
            f"({' or '.join(NAME_COLUMNS)}) and a free energy column in kcal/mol "
            f"({' or '.join(VALUE_COLUMNS)}); found {header}"
        )
    uncertainty_column = _column(header, UNCERTAINTY_COLUMNS)

    measurements: dict[str, Measurement] = {}
    for row in reader:
        name = (row.get(name_column) or "").strip()
        raw = (row.get(value_column) or "").strip()
        if not name or not raw:
            continue
        if name in measurements:
            raise InputValidationError(f"{path}: ligand '{name}' appears more than once")

        uncertainty: float | None = None
        if uncertainty_column is not None:
            raw_uncertainty = (row.get(uncertainty_column) or "").strip()
            if raw_uncertainty:
                uncertainty = _number(raw_uncertainty, name, uncertainty_column, path)

        measurements[name] = Measurement(
            name=name, dg=_number(raw, name, value_column, path), uncertainty=uncertainty
        )

    if not measurements:
        raise InputValidationError(f"{path}: no rows with both a ligand name and a value")

    logger.info("Read %d measured value(s) from %s", len(measurements), path)
    return measurements
