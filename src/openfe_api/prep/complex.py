"""Preparing one complex: protein, ligand and cofactors, with a record of what happened."""

from __future__ import annotations

from pathlib import Path

from openfe_api.exceptions import InputValidationError
from openfe_api.log import get_logger
from openfe_api.prep import checks
from openfe_api.prep.ligand import build_from_spec, build_molecule, write_sdf
from openfe_api.prep.protein import (
    check_parameterizable,
    heavy_atom_positions,
    prepare_protein,
    protein_report,
)
from openfe_api.prep.report import ComplexPrepReport, MoleculeReport
from openfe_api.prep.structure import StructureCache
from openfe_api.provenance import collect_provenance
from openfe_api.schema.abfe import ComplexSpec

__all__ = ["prepare_complex"]

logger = get_logger(__name__)


def prepare_complex(
    spec: ComplexSpec,
    output_dir: Path,
    neutralize_ligands: bool = False,
    check_parameters: bool = True,
) -> ComplexPrepReport:
    """Prepare every component of one complex and write the results.

    Args:
        spec: The complex to prepare.
        output_dir: Directory to write prepared files into. It is created if needed.
        neutralize_ligands: Whether a charged ligand may be neutralized, which changes
            the molecule chemically.
        check_parameters: Whether to check that the prepared protein parameterizes with
            the Amber force field before any simulation is planned.

    Returns:
        The preparation report, also written to ``prep_report.json``.

    Raises:
        InputValidationError: If any component fails preparation or validation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = StructureCache()
    warnings: list[str] = []
    excluded: set[int] = set()

    ligand_name = spec.ligand_name()
    ligand, ligand_source, ligand_selector, ligand_residue = build_from_spec(
        spec.ligand, ligand_name, spec.structure, cache, neutralize_ligands
    )
    warnings.extend(ligand.warnings)
    checks.check_elements(ligand.molecule, ligand_name)
    checks.check_radicals(ligand.molecule, ligand_name)
    checks.check_net_charge(ligand.net_charge, ligand_name)
    if ligand_residue is not None:
        excluded.add(ligand_residue)

    copies: list[MoleculeReport] = []
    if ligand_residue is not None and spec.structure is not None:
        structure = cache.get(spec.structure)
        found = structure.find_copies(structure.residue(ligand_residue))
        for copy in found:
            excluded.add(copy.index)
            if spec.extra_ligand_copies == "drop":
                warnings.append(
                    f"{ligand_name}: dropped an additional copy of the ligand at "
                    f"{copy.describe()}, as requested by 'extra_ligand_copies: drop'."
                )
                continue

            copy_name = f"{ligand_name}_copy_{copy.chain_id}"
            prepared_copy = build_molecule(
                spec.ligand.smiles,
                structure.residue_pdb_block(copy),
                copy_name,
                neutralize_ligands,
            )
            checks.check_elements(prepared_copy.molecule, copy_name)
            copy_path = output_dir / f"{copy_name}.sdf"
            write_sdf(prepared_copy.molecule, copy_path)
            copies.append(
                MoleculeReport(
                    name=copy_name,
                    role="ligand_copy",
                    source=spec.structure,
                    selector=copy.describe(),
                    path=copy_path,
                    smiles=prepared_copy.smiles,
                    net_charge=prepared_copy.net_charge,
                    n_atoms=prepared_copy.molecule.GetNumAtoms(),
                    neutralized=prepared_copy.neutralized,
                )
            )
            warnings.append(
                f"{ligand_name}: kept an additional copy of the ligand at "
                f"{copy.describe()} as a non-alchemical component, as requested by "
                "'extra_ligand_copies: keep'."
            )

    cofactor_reports: list[MoleculeReport] = []
    for cofactor_spec in spec.cofactors:
        cofactor, source, selector, residue_index = build_from_spec(
            cofactor_spec, cofactor_spec.name, spec.structure, cache, neutralize=False
        )
        warnings.extend(cofactor.warnings)
        checks.check_elements(cofactor.molecule, cofactor_spec.name)
        checks.check_radicals(cofactor.molecule, cofactor_spec.name)
        if residue_index is not None:
            excluded.add(residue_index)

        cofactor_path = output_dir / f"cofactor_{cofactor_spec.name}.sdf"
        write_sdf(cofactor.molecule, cofactor_path)
        cofactor_reports.append(
            MoleculeReport(
                name=cofactor_spec.name,
                role="cofactor",
                source=source,
                selector=selector,
                path=cofactor_path,
                smiles=cofactor.smiles,
                net_charge=cofactor.net_charge,
                n_atoms=cofactor.molecule.GetNumAtoms(),
            )
        )

    protein_source = spec.protein.path or spec.structure
    if protein_source is None:
        raise InputValidationError(f"{spec.name}: no source for the protein")

    protein_structure = cache.get(protein_source)
    protein = prepare_protein(
        protein_structure,
        output_dir / "protein.pdb",
        chains=spec.protein.chains,
        ph=spec.protein.ph,
        keep_waters=spec.protein.keep_waters,
        exclude_residues=excluded if protein_source == spec.structure else None,
    )
    warnings.extend(protein.warnings)

    if check_parameters:
        check_parameterizable(protein.path)

    warnings.extend(
        checks.check_clashes(
            ligand.molecule, heavy_atom_positions(protein.path), ligand_name, "the protein"
        )
    )

    ligand_path = output_dir / "ligand.sdf"
    write_sdf(ligand.molecule, ligand_path)

    report = ComplexPrepReport(
        name=spec.name,
        directory=output_dir,
        protein=protein_report(protein_source, protein, check_parameters),
        ligand=MoleculeReport(
            name=ligand_name,
            role="ligand",
            source=ligand_source,
            selector=ligand_selector,
            path=ligand_path,
            smiles=ligand.smiles,
            net_charge=ligand.net_charge,
            n_atoms=ligand.molecule.GetNumAtoms(),
            neutralized=ligand.neutralized,
        ),
        ligand_copies=copies,
        cofactors=cofactor_reports,
        warnings=warnings,
        provenance=collect_provenance(),
    )
    report.write(output_dir / "prep_report.json")
    logger.info("Prepared complex '%s' in %s", spec.name, output_dir)
    return report
