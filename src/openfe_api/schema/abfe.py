"""The absolute binding free energy input contract.

Each complex is independent: one protein, one alchemical ligand, and any cofactors that are
present unchanged in both end states.
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
    named_paths,
)

__all__ = ["AbfeRequest", "ComplexSpec", "LigandSpec"]


class LigandSpec(MoleculeSpec):
    """The alchemical ligand of a complex.

    Attributes:
        name: Optional ligand name. Defaults to the complex name during preparation.
    """

    name: str | None = Field(default=None, pattern=NAME_PATTERN)


class ComplexSpec(StrictModel):
    """One protein-ligand system to run.

    Attributes:
        name: Unique name for the complex, used for directories and result labels.
        structure: Combined structure file holding protein, ligand and cofactors, as
            produced by AF3 or Boltz-2. Components may instead supply their own ``path``.
        protein: Protein specification.
        ligand: Alchemical ligand specification.
        cofactors: Cofactors present in both end states.
        extra_ligand_copies: What to do with further copies of the ligand found in a
            combined structure file. ``drop`` removes them; ``keep`` retains them as
            non-alchemical components. Required whenever the ligand comes from a combined
            structure file, and rejected otherwise. There is no default: a homodimer with
            two ligand copies is never resolved by guessing.
    """

    name: str = Field(pattern=NAME_PATTERN)
    structure: Path | None = None
    protein: ProteinSpec = Field(default_factory=ProteinSpec)
    ligand: LigandSpec
    cofactors: list[CofactorSpec] = Field(default_factory=list)
    extra_ligand_copies: Literal["drop", "keep"] | None = None

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
    def _check_sources(self) -> ComplexSpec:
        if self.protein.path is None and self.structure is None:
            raise ValueError(
                "no source for the protein: set 'protein.path' or the complex's 'structure'"
            )
        if self.ligand.path is None and self.structure is None:
            raise ValueError(
                "no source for the ligand: set 'ligand.path' or the complex's 'structure'"
            )
        for cofactor in self.cofactors:
            if cofactor.path is None and self.structure is None:
                raise ValueError(
                    f"no source for cofactor '{cofactor.name}': set its 'path' or the "
                    "complex's 'structure'"
                )

        if self.ligand.path is None and self.ligand.selector is None:
            raise ValueError(
                "'ligand.selector' is required when the ligand comes from the combined "
                "'structure' file, so the alchemical molecule is unambiguous"
            )
        for cofactor in self.cofactors:
            if cofactor.path is None and cofactor.selector is None:
                raise ValueError(
                    f"'selector' is required for cofactor '{cofactor.name}' when it comes "
                    "from the combined 'structure' file"
                )

        names = [cofactor.name for cofactor in self.cofactors]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate cofactor names: {', '.join(duplicates)}")

        return self

    @model_validator(mode="after")
    def _check_extra_ligand_copies(self) -> ComplexSpec:
        from_structure = self.ligand.path is None
        if from_structure and self.extra_ligand_copies is None:
            raise ValueError(
                "'extra_ligand_copies' is required (no default) when the ligand comes from "
                "a combined 'structure' file: set it to 'drop' to remove further copies of "
                "the ligand, or 'keep' to retain them as non-alchemical components"
            )
        if not from_structure and self.extra_ligand_copies is not None:
            raise ValueError(
                "'extra_ligand_copies' only applies to a ligand read from a combined "
                "'structure' file; remove it when 'ligand.path' is set"
            )
        return self

    def ligand_name(self) -> str:
        """Return the ligand's name, falling back to the complex name.

        Returns:
            The explicit ``ligand.name`` if set, otherwise ``name``.
        """
        return self.ligand.name or self.name

    def path_holders(self) -> list[PathHolder]:
        """Return every attribute of this complex that holds an input path.

        Returns:
            Triples of holder, attribute name, and the complex name as the error label.
        """
        holders: list[PathHolder] = [
            (self, "structure", self.name),
            (self.protein, "path", self.name),
            (self.ligand, "path", self.name),
        ]
        holders.extend((cofactor, "path", self.name) for cofactor in self.cofactors)
        return holders

    def input_paths(self) -> list[Path]:
        """Return every input file this complex refers to.

        Returns:
            Paths in declaration order, without duplicates.
        """
        return named_paths(self.path_holders())


class AbfeRequest(RequestBase):
    """A complete, validated description of an ABFE campaign.

    Attributes:
        protocol: Free energy protocol to run.
        complexes: Systems to run, with unique names.
        neutralize_ligands: Whether preparation may neutralize a charged ligand.
            ``AbsoluteBindingProtocol`` rejects charged ligands, but neutralizing changes
            the molecule chemically, so this is off by default and warns loudly when used.
    """

    protocol: Literal["abfe"]
    complexes: list[ComplexSpec] = Field(min_length=1)
    neutralize_ligands: bool = False

    @model_validator(mode="after")
    def _check_unique_complex_names(self) -> AbfeRequest:
        names = [complex_spec.name for complex_spec in self.complexes]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate complex names: {', '.join(duplicates)}")
        return self

    def initial_run_names(self) -> list[str]:
        """Return the complex names, which are this protocol's runs.

        Returns:
            Complex names in declaration order.
        """
        return [complex_spec.name for complex_spec in self.complexes]

    def path_holders(self) -> list[PathHolder]:
        """Return every attribute across all complexes that holds an input path.

        Returns:
            Triples of holder, attribute name, and the owning complex's name.
        """
        holders: list[PathHolder] = []
        for complex_spec in self.complexes:
            holders.extend(complex_spec.path_holders())
        return holders
