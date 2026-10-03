"""Preparing one system for plain MD: whatever molecules share the box.

Nothing here is alchemical, so a system may hold a protein, several ligands, cofactors, or
only one of those, and a charged molecule is no problem.
"""

from __future__ import annotations

from pathlib import Path

from openfe_api.exceptions import InputValidationError
from openfe_api.log import get_logger
from openfe_api.prep import checks
from openfe_api.prep.ligand import build_from_spec, write_sdf
from openfe_api.prep.protein import (
    check_parameterizable,
    heavy_atom_positions,
    prepare_protein,
    protein_report,
)
from openfe_api.prep.report import MoleculeReport, ProteinReport, SystemPrepReport
from openfe_api.prep.structure import StructureCache
from openfe_api.provenance import collect_provenance
from openfe_api.schema.md import MdSystemSpec

__all__ = ["prepare_system"]

logger = get_logger(__name__)


def prepare_system(
    spec: MdSystemSpec,
    output_dir: Path,
    check_parameters: bool = True,
) -> SystemPrepReport:
    """Prepare every molecule of one system and write the results.

    Args:
        spec: The system to prepare.
        output_dir: Directory to write prepared files into. It is created if needed.
        check_parameters: Whether to check that the prepared protein parameterizes with the
            Amber force field before any simulation is planned.

    Returns:
        The preparation report, also written to ``prep_report.json``.

    Raises:
        InputValidationError: If any molecule fails preparation or validation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = StructureCache()
    warnings: list[str] = []
    excluded: set[int] = set()
    molecules: list[MoleculeReport] = []
    prepared_molecules = []

    for role, entries in (("ligand", spec.ligands), ("cofactor", spec.cofactors)):
        for entry in entries:
            prepared, source, selector, residue_index = build_from_spec(
                entry, entry.name, spec.structure, cache
            )
            warnings.extend(prepared.warnings)
            checks.check_elements(prepared.molecule, entry.name)
            checks.check_radicals(prepared.molecule, entry.name)
            if residue_index is not None:
                excluded.add(residue_index)

            path = output_dir / f"{entry.name}.sdf"
            write_sdf(prepared.molecule, path)
            molecules.append(
                MoleculeReport(
                    name=entry.name,
                    role=role,
                    source=source,
                    selector=selector,
                    path=path,
                    smiles=prepared.smiles,
                    net_charge=prepared.net_charge,
                    n_atoms=prepared.molecule.GetNumAtoms(),
                )
            )
            prepared_molecules.append((entry.name, prepared))

    record: ProteinReport | None = None
    if spec.protein is not None:
        protein_source = spec.protein.path or spec.structure
        if protein_source is None:
            raise InputValidationError(f"{spec.name}: no source for the protein")

        protein = prepare_protein(
            cache.get(protein_source),
            output_dir / "protein.pdb",
            chains=spec.protein.chains,
            ph=spec.protein.ph,
            keep_waters=spec.protein.keep_waters,
            exclude_residues=excluded if protein_source == spec.structure else None,
        )
        warnings.extend(protein.warnings)
        if check_parameters:
            check_parameterizable(protein.path)

        pocket = heavy_atom_positions(protein.path)
        for name, prepared in prepared_molecules:
            warnings.extend(checks.check_clashes(prepared.molecule, pocket, name, "the protein"))

        record = protein_report(protein_source, protein, check_parameters)

    report = SystemPrepReport(
        name=spec.name,
        directory=output_dir,
        protein=record,
        molecules=molecules,
        warnings=warnings,
        provenance=collect_provenance(),
    )
    report.write(output_dir / "prep_report.json")
    logger.info(
        "Prepared system '%s': %s, %d molecule(s)",
        spec.name,
        "protein present" if record is not None else "no protein",
        len(molecules),
    )
    return report
