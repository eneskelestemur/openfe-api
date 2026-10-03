"""Records of what preparation did to a complex."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ComplexPrepReport",
    "LigandPlacement",
    "MoleculeReport",
    "ProteinReport",
    "SeriesPrepReport",
    "SystemPrepReport",
]


class MoleculeReport(BaseModel):
    """What preparation produced for one small molecule.

    Attributes:
        name: Molecule name.
        role: ``ligand``, ``ligand_copy`` or ``cofactor``.
        source: File the pose was read from.
        selector: Selector used inside the source file, if any.
        path: Prepared SDF file.
        smiles: Canonical SMILES of the prepared molecule.
        net_charge: Net formal charge.
        n_atoms: Atom count including hydrogens.
        neutralized: Whether the molecule was neutralized.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    source: Path
    selector: str | None = None
    path: Path
    smiles: str
    net_charge: int
    n_atoms: int
    neutralized: bool = False


class ProteinReport(BaseModel):
    """What preparation produced for the protein.

    Attributes:
        source: File the protein was read from.
        path: Prepared PDB file.
        chains: Chain identifiers kept.
        n_atoms: Atom count after preparation.
        n_residues: Residue count after preparation.
        dropped_residues: Residues removed during preparation.
        parameterized: Whether the force field check ran and passed.
    """

    model_config = ConfigDict(extra="forbid")

    source: Path
    path: Path
    chains: list[str]
    n_atoms: int
    n_residues: int
    dropped_residues: list[str] = Field(default_factory=list)
    parameterized: bool = False


class ComplexPrepReport(BaseModel):
    """The full record of preparing one complex.

    Attributes:
        name: Complex name.
        prepared_at: When preparation finished.
        directory: Directory holding the prepared files.
        protein: The protein record.
        ligand: The alchemical ligand record.
        ligand_copies: Records for retained copies of the ligand.
        cofactors: Records for the cofactors.
        warnings: Everything the user should look at before running.
        provenance: Versions of the tools used.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    prepared_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    directory: Path
    protein: ProteinReport
    ligand: MoleculeReport
    ligand_copies: list[MoleculeReport] = Field(default_factory=list)
    cofactors: list[MoleculeReport] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)

    def write(self, path: Path) -> None:
        """Write the report as JSON.

        Args:
            path: File to write to. Parent directories are created as needed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


class LigandPlacement(BaseModel):
    """How one ligand of a series was placed into the reference frame.

    Attributes:
        name: Ligand name.
        source: File the ligand came from, or ``smiles`` when it had none.
        path: Prepared SDF file.
        smiles: Canonical SMILES of the prepared molecule.
        net_charge: Net formal charge.
        pose: How the pose was obtained: ``keep`` for the supplied coordinates, ``mcs`` for
            a pose built on the common core, ``shape`` for one overlaid on the reference
            ligand by Open3DAlign. Never ``auto``: the resolved path is recorded.
        core_smarts: The common core with the reference ligand, absent when it shares none.
        core_size: Heavy atoms in that core.
        core_fraction: Core size as a fraction of the smaller molecule's heavy atoms.
        perturbed_atoms: Heavy atoms outside the core, across both molecules.
        scaffold: Murcko scaffold SMILES.
        core_rmsd: Core RMSD to the reference ligand after placement, in angstrom.
        shape_score: Open3DAlign score against the reference ligand, for a ``shape`` pose.
            Higher is a better overlap, on a scale that grows with molecule size.
        shape_rmsd: RMSD in angstrom of the atoms Open3DAlign matched.
        clashes: Heavy atoms within the clash cutoff of a protein atom.
        worst_contact: Shortest heavy-atom distance to a protein atom, in angstrom.
        superposition_rmsd: Backbone RMSD of this ligand's own structure onto the
            reference, when it brought one.
        predicted_core_rmsd: Core RMSD between the generated pose and this ligand's own
            predicted pose, when it had one. A large value means the prediction disagrees
            with the series' binding mode, or the core matched the wrong atoms.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    source: str
    path: Path
    smiles: str
    net_charge: int
    pose: str
    scaffold: str
    core_smarts: str | None = None
    core_size: int | None = None
    core_fraction: float | None = None
    perturbed_atoms: int | None = None
    core_rmsd: float | None = None
    shape_score: float | None = None
    shape_rmsd: float | None = None
    clashes: int = 0
    worst_contact: float | None = None
    superposition_rmsd: float | None = None
    predicted_core_rmsd: float | None = None


class SeriesPrepReport(BaseModel):
    """The full record of preparing one series of ligands in one receptor.

    Attributes:
        name: Campaign name.
        prepared_at: When preparation finished.
        directory: Directory holding the prepared files.
        reference: Name of the ligand supplying the protein and the trusted pose.
        protein: The protein record.
        ligands: One record per ligand that was placed, the reference first.
        dropped: Why each excluded ligand was excluded, keyed by name.
        cofactors: Records for the cofactors.
        warnings: Everything the user should look at before running.
        provenance: Versions of the tools used.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    prepared_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    directory: Path
    reference: str
    protein: ProteinReport
    ligands: list[LigandPlacement] = Field(default_factory=list)
    dropped: dict[str, str] = Field(default_factory=dict)
    cofactors: list[MoleculeReport] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)

    def write(self, path: Path) -> None:
        """Write the report as JSON.

        Args:
            path: File to write to. Parent directories are created as needed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


class SystemPrepReport(BaseModel):
    """The full record of preparing one system for plain MD.

    Nothing here is alchemical, so the molecules are one list rather than a named ligand and
    its cofactors, and the protein is optional.

    Attributes:
        name: System name.
        prepared_at: When preparation finished.
        directory: Directory holding the prepared files.
        protein: The protein record, absent for a system without one.
        molecules: One record per small molecule, ligands before cofactors.
        warnings: Everything the user should look at before running.
        provenance: Versions of the tools used.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    prepared_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    directory: Path
    protein: ProteinReport | None = None
    molecules: list[MoleculeReport] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)

    def write(self, path: Path) -> None:
        """Write the report as JSON.

        Args:
            path: File to write to. Parent directories are created as needed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
