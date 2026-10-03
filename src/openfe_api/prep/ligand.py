"""Building a simulation-ready molecule from a SMILES and a pose."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

from openfe_api.exceptions import InputValidationError
from openfe_api.log import get_logger
from openfe_api.prep.structure import StructureCache
from openfe_api.schema.common import MoleculeSpec

__all__ = [
    "PreparedMolecule",
    "build_from_file",
    "build_from_smiles",
    "build_from_spec",
    "build_molecule",
    "parse_template",
    "read_single_molecule",
    "unspecified_stereo",
    "write_sdf",
]

logger = get_logger(__name__)


@dataclass
class PreparedMolecule:
    """A molecule built from a SMILES and a pose.

    Attributes:
        molecule: The RDKit molecule, with explicit hydrogens and 3D coordinates.
        name: Name assigned to the molecule.
        smiles: Canonical isomeric SMILES of the prepared molecule.
        net_charge: Net formal charge.
        neutralized: Whether neutralization was applied to reach a neutral form.
        warnings: Warnings raised while building the molecule.
    """

    molecule: Chem.Mol
    name: str
    smiles: str
    net_charge: int
    neutralized: bool = False
    warnings: list[str] = field(default_factory=list)


def parse_template(smiles: str, name: str) -> Chem.Mol:
    """Parse a reference SMILES.

    Stereochemistry may be left undefined. An undefined center means the stereochemistry
    is unknown, as when a racemic mixture is passed to a structure model, and is resolved
    from the predicted pose rather than rejected.

    Args:
        smiles: The reference SMILES.
        name: Molecule name, used in error messages.

    Returns:
        The parsed molecule, without explicit hydrogens.

    Raises:
        InputValidationError: If the SMILES cannot be parsed.
    """
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise InputValidationError(f"{name}: SMILES could not be parsed by RDKit: '{smiles}'")
    return template


def unspecified_stereo(molecule: Chem.Mol) -> tuple[set[int], set[int]]:
    """Find the stereo centers a molecule leaves undefined.

    Args:
        molecule: The molecule to inspect.

    Returns:
        A tuple of the atom indices with undefined chirality and the bond indices with
        undefined double bond stereochemistry.
    """
    atoms: set[int] = set()
    bonds: set[int] = set()
    for element in Chem.FindPotentialStereo(molecule):
        if element.specified != Chem.StereoSpecified.Unspecified:
            continue
        if element.type == Chem.StereoType.Atom_Tetrahedral:
            atoms.add(element.centeredOn)
        elif element.type == Chem.StereoType.Bond_Double:
            bonds.add(element.centeredOn)
    return atoms, bonds


def _clear_stereo(
    molecule: Chem.Mol,
    atoms: set[int],
    bonds: set[int],
) -> Chem.Mol:
    """Return a copy of a molecule with stereochemistry cleared at given positions."""
    copy = Chem.Mol(molecule)
    for index in atoms:
        copy.GetAtomWithIdx(index).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for index in bonds:
        copy.GetBondWithIdx(index).SetStereo(Chem.BondStereo.STEREONONE)
    return copy


def _map_to_pose(template: Chem.Mol, assigned: Chem.Mol, name: str) -> tuple[int, ...]:
    """Map template atom indices onto the posed molecule's atom indices."""
    flat = Chem.Mol(template)
    Chem.RemoveStereochemistry(flat)
    match = assigned.GetSubstructMatch(flat)
    if not match:
        raise InputValidationError(
            f"{name}: the SMILES and the pose could not be aligned atom by atom, so their "
            "stereochemistry cannot be compared."
        )
    return match


def _describe_center(molecule: Chem.Mol, atom_index: int) -> str:
    """Describe an atom's assigned chirality."""
    atom = molecule.GetAtomWithIdx(atom_index)
    code = atom.GetPropsAsDict().get("_CIPCode", "?")
    return f"{atom.GetSymbol()}{atom_index} ({code})"


def _comparable_smiles(molecule: Chem.Mol) -> str:
    """Return a canonical SMILES for comparing a pose against its reference SMILES.

    Phosphorus and sulfur carrying two or more terminal oxygens lose their stereochemistry
    first. Those centers are not stereogenic -- a phosphate's oxygens are equivalent by
    resonance -- so RDKit reads a chirality from the coordinates that the SMILES assigns
    arbitrarily, and it would read as a mismatch. Sulfoxides keep theirs.
    """
    copy = Chem.Mol(molecule)
    for atom in copy.GetAtoms():
        if atom.GetSymbol() not in {"P", "S"}:
            continue
        terminal_oxygens = sum(
            1
            for neighbor in atom.GetNeighbors()
            if neighbor.GetSymbol() == "O" and neighbor.GetDegree() == 1
        )
        if terminal_oxygens >= 2:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return Chem.MolToSmiles(copy)


def _neutralize(template: Chem.Mol, name: str) -> tuple[Chem.Mol, str]:
    """Neutralize a charged template molecule."""
    before = Chem.MolToSmiles(template)
    neutral = rdMolStandardize.Uncharger().uncharge(Chem.Mol(template))
    charge = Chem.GetFormalCharge(neutral)
    if charge != 0:
        raise InputValidationError(
            f"{name}: neutralization was requested but the molecule still carries a net "
            f"charge of {charge:+d}. Charges that are not removable by protonation, such "
            "as a quaternary nitrogen, cannot be neutralized. Supply a neutral analogue "
            "or use a protocol that supports charged ligands."
        )

    after = Chem.MolToSmiles(neutral)
    warning = (
        f"{name}: NEUTRALIZED. The molecule was changed from '{before}' to '{after}'. "
        "This is a different chemical species from the one you supplied, and the computed "
        "affinity refers to the neutral form."
    )
    return neutral, warning


def read_single_molecule(path: Path, name: str) -> Chem.Mol:
    """Read a molecule file that must hold exactly one molecule.

    Args:
        path: An SDF, MOL or MOL2 file.
        name: Molecule name, used in error messages.

    Returns:
        The molecule, with any hydrogens it already carried.

    Raises:
        InputValidationError: If the format is unsupported, the file cannot be read, or
            it holds no molecule or more than one.
    """
    suffix = path.suffix.lower()
    if suffix in {".sdf", ".mol"}:
        try:
            molecules = [
                molecule
                for molecule in Chem.SDMolSupplier(str(path), removeHs=False)
                if molecule is not None
            ]
        except OSError as error:
            raise InputValidationError(f"{name}: {path} could not be read: {error}") from error
    elif suffix == ".mol2":
        try:
            molecule = Chem.MolFromMol2File(str(path), removeHs=False)
        except OSError as error:
            raise InputValidationError(f"{name}: {path} could not be read: {error}") from error
        molecules = [molecule] if molecule is not None else []
    else:
        raise InputValidationError(
            f"{name}: unsupported molecule format '{suffix}' in {path}; use .sdf, .mol "
            "or .mol2, or select the molecule from a structure file"
        )

    if not molecules:
        raise InputValidationError(f"{name}: no readable molecule found in {path}")
    if len(molecules) > 1:
        raise InputValidationError(
            f"{name}: {path} holds {len(molecules)} molecules, but exactly one is "
            "required. Split the file so each molecule has its own entry in the request."
        )
    return molecules[0]


def _describe_undefined(molecule: Chem.Mol, atom_index: int) -> str:
    """Name an atom whose configuration the SMILES leaves open."""
    atom = molecule.GetAtomWithIdx(atom_index)
    neighbors = ",".join(sorted(n.GetSymbol() for n in atom.GetNeighbors()))
    return f"atom {atom_index} ({atom.GetSymbol()} bonded to {neighbors})"


def build_from_smiles(smiles: str, name: str, neutralize: bool = False) -> PreparedMolecule:
    """Build a molecule from a SMILES alone, leaving it without coordinates.

    Undefined stereocenters are rejected rather than adopted: there is no pose to read them
    from, and enumerating them would silently multiply the campaign.

    Args:
        smiles: The SMILES defining the molecule.
        name: Name to assign to the molecule.
        neutralize: Whether to neutralize a charged molecule.

    Returns:
        The prepared molecule, with explicit hydrogens and no conformer.

    Raises:
        InputValidationError: If the SMILES cannot be parsed, leaves stereochemistry
            undefined, or cannot be neutralized when asked.
    """
    template = parse_template(smiles, name)
    warnings: list[str] = []
    neutralized = False

    if neutralize and Chem.GetFormalCharge(template) != 0:
        template, warning = _neutralize(template, name)
        warnings.append(warning)
        neutralized = True
        logger.warning(warning)

    undefined_atoms, undefined_bonds = unspecified_stereo(template)
    if undefined_atoms or undefined_bonds:
        problems: list[str] = []
        if undefined_atoms:
            listed = ", ".join(
                sorted(_describe_undefined(template, index) for index in undefined_atoms)
            )
            problems.append(f"{len(undefined_atoms)} undefined stereocenter(s): {listed}")
        if undefined_bonds:
            problems.append(f"{len(undefined_bonds)} double bond(s) without defined geometry")
        raise InputValidationError(
            f"{name}: the SMILES has {' and '.join(problems)}, and supplies no pose to take "
            "them from. Give the SMILES a defined configuration at every center, or supply "
            "coordinates for this ligand so the configuration can be read from them."
        )

    molecule = Chem.AddHs(template)
    molecule.SetProp("_Name", name)
    Chem.SanitizeMol(molecule)

    return PreparedMolecule(
        molecule=molecule,
        name=name,
        smiles=Chem.MolToSmiles(template),
        net_charge=Chem.GetFormalCharge(molecule),
        neutralized=neutralized,
        warnings=warnings,
    )


def build_from_file(
    smiles: str,
    path: Path,
    name: str,
    neutralize: bool = False,
) -> PreparedMolecule:
    """Build a molecule from a dedicated molecule file and a reference SMILES.

    The file already carries bond orders, but it is still checked against the SMILES, so
    a mislabeled or edited file cannot slip through.

    Args:
        smiles: Reference SMILES defining the chemistry.
        path: An SDF, MOL or MOL2 file holding exactly one molecule.
        name: Name to assign to the molecule.
        neutralize: Whether to neutralize a charged molecule.

    Returns:
        The prepared molecule.

    Raises:
        InputValidationError: If the file cannot be read, or its molecule disagrees with
            the SMILES on composition, connectivity or stereochemistry.
    """
    molecule = read_single_molecule(path, name)
    if molecule.GetNumConformers() == 0:
        raise InputValidationError(f"{name}: {path} has no 3D coordinates")

    return _finalize(molecule, smiles, name, neutralize, source=str(path))


def build_molecule(
    smiles: str,
    pdb_block: str,
    name: str,
    neutralize: bool = False,
) -> PreparedMolecule:
    """Build a molecule by applying a reference SMILES to a pose.

    The pose supplies coordinates only. Bond orders and formal charges come from the
    SMILES, stereochemistry is read back from the 3D coordinates and checked against the
    SMILES, and hydrogens are added last.

    Args:
        smiles: Reference SMILES defining the chemistry.
        pdb_block: PDB text holding the molecule's heavy atoms.
        name: Name to assign to the molecule.
        neutralize: Whether to neutralize a charged molecule. This changes the molecule
            chemically and is reported as a warning.

    Returns:
        The prepared molecule.

    Raises:
        InputValidationError: If the SMILES cannot be parsed or is stereochemically
            incomplete, the pose cannot be read, the pose and the SMILES disagree on
            heavy atom count or connectivity, or their stereochemistry differs.
    """
    pose = Chem.MolFromPDBBlock(pdb_block, removeHs=False, proximityBonding=True)
    if pose is None:
        raise InputValidationError(
            f"{name}: the pose could not be read as a molecule. Check that the selected "
            "residue holds a complete small molecule."
        )

    return _finalize(pose, smiles, name, neutralize, source="the pose")


def _finalize(
    pose: Chem.Mol,
    smiles: str,
    name: str,
    neutralize: bool,
    source: str,
) -> PreparedMolecule:
    """Apply a reference SMILES to a posed molecule and check the two agree."""
    template = parse_template(smiles, name)
    warnings: list[str] = []
    neutralized = False

    if neutralize and Chem.GetFormalCharge(template) != 0:
        template, warning = _neutralize(template, name)
        warnings.append(warning)
        neutralized = True
        logger.warning(warning)

    heavy_pose = Chem.RemoveHs(pose)
    if heavy_pose.GetNumAtoms() != template.GetNumAtoms():
        raise InputValidationError(
            f"{name}: {source} has {heavy_pose.GetNumAtoms()} heavy atoms but the SMILES "
            f"has {template.GetNumAtoms()}. The SMILES must describe exactly the molecule "
            "in the structure file, including any part that may have been omitted."
        )

    try:
        assigned = AllChem.AssignBondOrdersFromTemplate(template, heavy_pose)
    except (ValueError, RuntimeError) as error:
        raise InputValidationError(
            f"{name}: the SMILES could not be matched onto {source} ({error}). The heavy "
            "atom count agrees, so the connectivity differs from the SMILES. Check that "
            "the SMILES is the right molecule and that the pose geometry is reasonable."
        ) from error

    Chem.AssignStereochemistryFrom3D(assigned)

    undefined_atoms, undefined_bonds = unspecified_stereo(template)
    if undefined_atoms or undefined_bonds:
        mapping = _map_to_pose(template, assigned, name)
        pose_atoms = {mapping[index] for index in undefined_atoms}
        pose_bonds = {
            assigned.GetBondBetweenAtoms(
                mapping[template.GetBondWithIdx(index).GetBeginAtomIdx()],
                mapping[template.GetBondWithIdx(index).GetEndAtomIdx()],
            ).GetIdx()
            for index in undefined_bonds
        }
        adopted = ", ".join(sorted(_describe_center(assigned, index) for index in pose_atoms))
        if adopted:
            warnings.append(
                f"{name}: the SMILES left {len(pose_atoms)} stereocenter(s) undefined; the "
                f"configuration was taken from the pose: {adopted}."
            )
        if pose_bonds:
            warnings.append(
                f"{name}: the SMILES left {len(pose_bonds)} double bond(s) without defined "
                "geometry; the configuration was taken from the pose."
            )
    else:
        pose_atoms, pose_bonds = set(), set()

    pose_smiles = _comparable_smiles(_clear_stereo(assigned, pose_atoms, pose_bonds))
    template_smiles = _comparable_smiles(_clear_stereo(template, undefined_atoms, undefined_bonds))
    if pose_smiles != template_smiles:
        raise InputValidationError(
            f"{name}: the stereochemistry of {source} does not match the SMILES.\n"
            f"  from the pose:   {_comparable_smiles(assigned)}\n"
            f"  from the SMILES: {_comparable_smiles(template)}\n"
            "Stereochemistry the SMILES defines must be reproduced by the pose; structure "
            "models can invert stereocenters. Correct the SMILES or drop this pose."
        )

    molecule = Chem.AddHs(assigned, addCoords=True)
    molecule.SetProp("_Name", name)
    Chem.SanitizeMol(molecule)

    return PreparedMolecule(
        molecule=molecule,
        name=name,
        smiles=Chem.MolToSmiles(Chem.RemoveHs(molecule)),
        net_charge=Chem.GetFormalCharge(molecule),
        neutralized=neutralized,
        warnings=warnings,
    )


def write_sdf(molecule: Chem.Mol, path: Path) -> None:
    """Write a molecule to an SDF file, creating parent directories as needed.

    Args:
        molecule: The molecule to write.
        path: File to write to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(path)) as writer:
        writer.write(molecule)


def build_from_spec(
    spec: MoleculeSpec,
    name: str,
    structure_path: Path | None,
    cache: StructureCache,
    neutralize: bool = False,
) -> tuple[PreparedMolecule, Path, str | None, int | None]:
    """Build one molecule from its own coordinate file or from a combined structure.

    Args:
        spec: The molecule's specification.
        name: Name to use in messages and in the report.
        structure_path: The combined structure file to read from when the molecule has no
            ``path`` of its own.
        cache: Cache of already-parsed structures.
        neutralize: Whether a charged molecule may be neutralized.

    Returns:
        The prepared molecule, the file it came from, the selector used inside that file if
        any, and the residue index it occupied if it came from a structure.

    Raises:
        InputValidationError: If the molecule has neither a path nor a resolvable selector,
            or it fails to build.
    """
    if spec.path is not None:
        return build_from_file(spec.smiles, spec.path, name, neutralize), spec.path, None, None

    if structure_path is None or spec.selector is None:
        raise InputValidationError(
            f"{name}: no coordinates to build from; set its 'path', or give the entry a "
            "'structure' file and this molecule a 'selector'"
        )

    structure = cache.get(structure_path)
    residue = structure.select_one_residue(spec.selector, name)
    prepared = build_molecule(spec.smiles, structure.residue_pdb_block(residue), name, neutralize)
    return prepared, structure_path, spec.selector.describe(), residue.index
