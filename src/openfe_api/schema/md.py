"""The plain molecular dynamics input contract.

A system is whatever should be simulated together: a protein-ligand complex, an apo protein,
or a ligand in water. Nothing here is alchemical, so there is no single-ligand rule and no
charge policy; any number of molecules may share one box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from openfe_api.schema.common import (
    NAME_PATTERN,
    STRUCTURE_SUFFIXES,
    CofactorSpec,
    MoleculeSpec,
    PathHolder,
    ProteinSpec,
    RequestBase,
    StrictModel,
)

__all__ = ["MdLigandSpec", "MdRequest", "MdSystemSpec"]


class MdLigandSpec(MoleculeSpec):
    """One ligand in a simulated system.

    Attributes:
        name: Ligand name, unique within its system.
    """

    name: str = Field(pattern=NAME_PATTERN)


class MdSystemSpec(StrictModel):
    """One system to simulate.

    At least one molecule is needed: a protein, a ligand, or a cofactor. An empty box has
    nothing to simulate.

    Attributes:
        name: Unique name for the system, used for directories and result labels.
        structure: Combined structure file holding the protein and any molecules read from
            it. Components may instead supply their own ``path``.
        protein: Protein specification, or None for a system without a protein.
        ligands: Ligands to simulate. Any number is allowed.
        cofactors: Cofactors to simulate alongside them. They are treated exactly as ligands
            are here and are named separately only to keep a request readable.
        solvent: Whether to solvate the system. Turning it off runs in vacuum, which needs a
            nonbonded method other than PME; planning sets one and says so.
    """

    name: str = Field(pattern=NAME_PATTERN)
    structure: Path | None = None
    protein: ProteinSpec | None = None
    ligands: list[MdLigandSpec] = Field(default_factory=list)
    cofactors: list[CofactorSpec] = Field(default_factory=list)
    solvent: bool = True

    @field_validator("structure")
    @classmethod
    def _check_structure_suffix(cls, value: Path | None) -> Path | None:
        if value is not None and value.suffix.lower() not in STRUCTURE_SUFFIXES:
            raise ValueError(
                f"'structure' must be a PDB or mmCIF file, got '{value.name}'; "
                f"supported suffixes: {', '.join(sorted(STRUCTURE_SUFFIXES))}"
            )
        return value

    @model_validator(mode="after")
    def _check_contents(self) -> MdSystemSpec:
        if self.protein is None and not self.ligands and not self.cofactors:
            raise ValueError(
                f"system '{self.name}' holds nothing to simulate; give it a 'protein', at "
                "least one entry in 'ligands', or a cofactor"
            )

        if self.protein is not None and self.protein.path is None and self.structure is None:
            raise ValueError(
                f"no source for the protein of system '{self.name}': set 'protein.path' or "
                "the system's 'structure'"
            )

        for molecule in self.molecules():
            if molecule.path is None and self.structure is None:
                raise ValueError(
                    f"no source for '{molecule.name}' in system '{self.name}': set its "
                    "'path' or the system's 'structure'"
                )
            if molecule.path is None and molecule.selector is None:
                raise ValueError(
                    f"'selector' is required for '{molecule.name}' in system '{self.name}' "
                    "when it comes from the combined 'structure' file"
                )

        names = [molecule.name for molecule in self.molecules()]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate molecule names in system '{self.name}': {', '.join(duplicates)}"
            )
        return self

    def molecules(self) -> list[MdLigandSpec | CofactorSpec]:
        """Return every small molecule in this system.

        Returns:
            The ligands followed by the cofactors, in declaration order.
        """
        return [*self.ligands, *self.cofactors]

    def path_holders(self) -> list[PathHolder]:
        """Return every attribute of this system that holds an input path.

        Returns:
            Triples of holder, attribute name, and the system name as the error label.
        """
        holders: list[PathHolder] = [(self, "structure", self.name)]
        if self.protein is not None:
            holders.append((self.protein, "path", self.name))
        holders.extend((molecule, "path", self.name) for molecule in self.molecules())
        return holders


class MdRequest(RequestBase):
    """A complete, validated description of a plain MD campaign.

    Attributes:
        protocol: Protocol to run.
        systems: Systems to simulate, with unique names.
    """

    protocol: Literal["md"]
    systems: list[MdSystemSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique_system_names(self) -> MdRequest:
        names = [system.name for system in self.systems]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate system names: {', '.join(duplicates)}")
        return self

    def initial_run_names(self) -> list[str]:
        """Return the system names, which are this protocol's runs.

        Returns:
            System names in declaration order.
        """
        return [system.name for system in self.systems]

    def path_holders(self) -> list[PathHolder]:
        """Return every attribute across all systems that holds an input path.

        Returns:
            Triples of holder, attribute name, and the owning system's name.
        """
        holders: list[PathHolder] = []
        for system in self.systems:
            holders.extend(system.path_holders())
        return holders
