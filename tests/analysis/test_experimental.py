"""Tests for reading a table of measured free energies."""

from __future__ import annotations

from pathlib import Path

import pytest

from openfe_api.analysis.experimental import read_measurements
from openfe_api.exceptions import InputValidationError


def _write(path: Path, text: str) -> Path:
    """Write a table and return its path."""
    path.write_text(text, encoding="utf-8")
    return path


def test_a_csv_is_read(tmp_path: Path) -> None:
    table = _write(tmp_path / "exp.csv", "name,dg\nlig_a,-9.2\nlig_b,-8.1\n")

    measurements = read_measurements(table)

    assert set(measurements) == {"lig_a", "lig_b"}
    assert measurements["lig_a"].dg == pytest.approx(-9.2)
    assert measurements["lig_a"].uncertainty is None


def test_a_tsv_is_read(tmp_path: Path) -> None:
    table = _write(tmp_path / "exp.tsv", "ligand\tdg (kcal/mol)\nlig_a\t-9.2\n")

    assert read_measurements(table)["lig_a"].dg == pytest.approx(-9.2)


def test_an_uncertainty_column_is_used(tmp_path: Path) -> None:
    table = _write(tmp_path / "exp.csv", "name,dg,uncertainty\nlig_a,-9.2,0.3\n")

    assert read_measurements(table)["lig_a"].uncertainty == pytest.approx(0.3)


def test_column_names_ignore_case_and_spacing(tmp_path: Path) -> None:
    table = _write(tmp_path / "exp.csv", " Name , Experimental \nlig_a,-9.2\n")

    assert read_measurements(table)["lig_a"].dg == pytest.approx(-9.2)


def test_rows_without_a_value_are_skipped(tmp_path: Path) -> None:
    table = _write(tmp_path / "exp.csv", "name,dg\nlig_a,-9.2\nlig_b,\n")

    assert set(read_measurements(table)) == {"lig_a"}


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="no measured values"):
        read_measurements(tmp_path / "absent.csv")


def test_a_table_without_the_columns_is_refused(tmp_path: Path) -> None:
    table = _write(tmp_path / "exp.csv", "compound,ic50\nlig_a,14.7\n")

    with pytest.raises(InputValidationError, match="needs a ligand name column"):
        read_measurements(table)


def test_a_value_that_is_not_a_number_is_refused(tmp_path: Path) -> None:
    table = _write(tmp_path / "exp.csv", "name,dg\nlig_a,strong\n")

    with pytest.raises(InputValidationError, match="not a number"):
        read_measurements(table)


def test_a_repeated_ligand_is_refused(tmp_path: Path) -> None:
    table = _write(tmp_path / "exp.csv", "name,dg\nlig_a,-9.2\nlig_a,-8.0\n")

    with pytest.raises(InputValidationError, match="more than once"):
        read_measurements(table)


def test_a_table_with_no_usable_rows_is_refused(tmp_path: Path) -> None:
    table = _write(tmp_path / "exp.csv", "name,dg\n,\n")

    with pytest.raises(InputValidationError, match="no rows"):
        read_measurements(table)
