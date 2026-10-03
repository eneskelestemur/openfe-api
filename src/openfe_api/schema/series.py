"""Input contract shared by the protocols that run a series of ligands in one receptor.

RBFE and SepTop both take one receptor, one reference pose, and a set of ligands placed into
that frame. Only their screening, their pose sources and their edge chemistry differ.
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

__all__ = [
    "AlignmentSpecBase",
    "SeriesLigandSpecBase",
    "SeriesRequestBase",
    "check_edges",
    "check_keep_has_coordinates",
    "check_network_names",
]


class SeriesLigandSpecBase(MoleculeSpec):
    """One ligand of a series, before a protocol narrows how its pose is obtained.

    A ligand arrives in its own combined structure file, as a dedicated coordinate file, or as
    SMILES alone, in which case its pose is generated.

    Attributes:
        name: Ligand name, unique within the campaign.
        structure: Combined structure file holding this ligand, and optionally the protein it
            was predicted against. Mutually exclusive with ``path``.
    """

    name: str = Field(pattern=NAME_PATTERN)
    structure: Path | None = None

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
    def _check_sources(self) -> SeriesLigandSpecBase:
        if self.path is not None and self.structure is not None:
            raise ValueError(
                f"ligand '{self.name}' sets both 'path' and 'structure'; use 'structure' "
                "with a 'selector' for a combined prediction file, or 'path' for a "
                "dedicated coordinate file"
            )
        if self.structure is not None and self.selector is None:
            raise ValueError(
                f"'selector' is required for ligand '{self.name}' because it comes from the "
                "combined 'structure' file, so the molecule is unambiguous"
            )
        return self

    def has_coordinates(self) -> bool:
        """Report whether this ligand brings a pose of its own.

        Returns:
            True if either a dedicated coordinate file or a combined structure file is set.
        """
        return self.path is not None or self.structure is not None


def check_keep_has_coordinates(spec: SeriesLigandSpecBase, pose: str | None) -> None:
    """Check that a ligand asking to keep its pose supplies one.

    Args:
        spec: The ligand.
        pose: The pose source set on it, if any.

    Raises:
        ValueError: If the ligand keeps a pose it does not supply.
    """
    if pose == "keep" and not spec.has_coordinates():
        raise ValueError(
            f"ligand '{spec.name}' sets 'pose: keep' but supplies no coordinates; set "
            "'path' or 'structure', or remove 'pose' to have one generated"
        )


def check_edges(edges: list[tuple[str, str]] | None, method: str) -> None:
    """Check an explicit edge list for self-edges.

    Args:
        edges: The ligand name pairs, if any were given.
        method: The network method they were given for, used in the message.

    Raises:
        ValueError: If a pair names the same ligand twice.
    """
    for first, second in edges or []:
        if first == second:
            raise ValueError(
                f"'network.edges' contains a self-edge for method '{method}': {first} to itself"
            )


def check_network_names(
    names: set[str], central_ligand: str | None, edges: list[tuple[str, str]] | None
) -> None:
    """Check that a network only names ligands the series holds.

    Args:
        names: Every ligand name in the series.
        central_ligand: The hub, if one was named.
        edges: The explicit edges, if any were given.

    Raises:
        ValueError: If the hub or an edge names a ligand that is not in the series.
    """
    if central_ligand is not None and central_ligand not in names:
        raise ValueError(
            f"'network.central_ligand' names '{central_ligand}', which is not one of the "
            f"ligands: {', '.join(sorted(names))}"
        )
    unknown = sorted({name for edge in edges or [] for name in edge if name not in names})
    if unknown:
        raise ValueError(
            f"'network.edges' names ligands that are not in the series: {', '.join(unknown)}"
        )


class AlignmentSpecBase(StrictModel):
    """How ligands are placed into the reference frame, apart from the pose source itself.

    Attributes:
        n_conformers: Conformers generated per ligand before one is chosen.
        max_core_rmsd: Largest core RMSD to the reference, in angstrom, that a generated
            conformer may have to be accepted.
        rank_by: How accepted conformers are ranked. ``clash`` prefers the pose fitting the
            reference pocket best; ``energy`` prefers the lowest force field energy.
        report_predicted_rmsd: Whether to superpose a ligand's own predicted complex and
            report the core RMSD to the generated pose. A large value means the prediction
            disagrees with the series' binding mode, or the core matched the wrong atoms.
        pocket_radius: Radius in angstrom around the reference ligand defining the pocket
            atoms used for superposition and clash scoring.
    """

    n_conformers: int = Field(default=50, ge=1)
    max_core_rmsd: float = Field(default=0.5, gt=0.0)
    rank_by: Literal["clash", "energy"] = "clash"
    report_predicted_rmsd: bool = True
    pocket_radius: float = Field(default=10.0, gt=0.0)


class SeriesRequestBase[LigandT: SeriesLigandSpecBase](RequestBase):
    """Fields and validation shared by every series request.

    Attributes:
        reference: Name of the ligand supplying the protein and the trusted pose. Defaults to
            the first entry in ``ligands``.
        protein: Protein specification, applied to the reference's structure.
        ligands: The series, with unique names. At least two are needed for one edge.
        cofactors: Cofactors present unchanged in both end states of every edge.
    """

    reference: str | None = Field(default=None, pattern=NAME_PATTERN)
    protein: ProteinSpec = Field(default_factory=ProteinSpec)
    ligands: list[LigandT] = Field(min_length=2)
    cofactors: list[CofactorSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_names(self) -> SeriesRequestBase[LigandT]:
        names = [ligand.name for ligand in self.ligands]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate ligand names: {', '.join(duplicates)}")

        cofactor_names = [cofactor.name for cofactor in self.cofactors]
        cofactor_duplicates = sorted(
            {name for name in cofactor_names if cofactor_names.count(name) > 1}
        )
        if cofactor_duplicates:
            raise ValueError(f"duplicate cofactor names: {', '.join(cofactor_duplicates)}")
        return self

    @model_validator(mode="after")
    def _check_reference(self) -> SeriesRequestBase[LigandT]:
        names = [ligand.name for ligand in self.ligands]
        if self.reference is not None and self.reference not in names:
            raise ValueError(
                f"'reference' names '{self.reference}', which is not one of the ligands: "
                f"{', '.join(names)}"
            )

        reference = self.reference_ligand()
        if not reference.has_coordinates():
            raise ValueError(
                f"the reference ligand '{reference.name}' must supply coordinates, because "
                "its pose defines the frame every other ligand is placed into; set its "
                "'structure' or 'path'"
            )
        if self.protein.path is None and reference.structure is None:
            raise ValueError(
                f"no source for the protein: the reference ligand '{reference.name}' must "
                "carry it in a combined 'structure' file, or 'protein.path' must be set"
            )
        return self

    @model_validator(mode="after")
    def _check_cofactor_sources(self) -> SeriesRequestBase[LigandT]:
        protein_structure = self.reference_ligand().structure
        for cofactor in self.cofactors:
            if cofactor.path is None and protein_structure is None:
                raise ValueError(
                    f"no source for cofactor '{cofactor.name}': set its 'path', since the "
                    "reference ligand has no combined 'structure' file to read it from"
                )
            if cofactor.path is None and cofactor.selector is None:
                raise ValueError(
                    f"'selector' is required for cofactor '{cofactor.name}' when it comes "
                    "from the reference's combined 'structure' file"
                )
        return self

    def reference_ligand(self) -> LigandT:
        """Return the ligand supplying the protein and the trusted pose.

        Returns:
            The ligand named by ``reference``, or the first entry if it is unset.
        """
        if self.reference is None:
            return self.ligands[0]
        return next(ligand for ligand in self.ligands if ligand.name == self.reference)

    def pose_source(self, ligand: LigandT) -> str:
        """Return how one ligand's pose is obtained.

        Args:
            ligand: The ligand to decide for.

        Returns:
            The pose source, which a protocol offering ``auto`` may return as such.

        Raises:
            NotImplementedError: If a subclass does not implement it.
        """
        raise NotImplementedError

    def requested_poses(self) -> dict[str, str]:
        """Return the pose source declared for every ligand, keyed by name.

        Returns:
            One entry per ligand, still holding ``auto`` where the protocol resolves it
            during preparation.
        """
        return {ligand.name: self.pose_source(ligand) for ligand in self.ligands}

    def ligand_names(self) -> set[str]:
        """Return every ligand name in the series.

        Returns:
            The names, unordered.
        """
        return {ligand.name for ligand in self.ligands}

    def initial_run_names(self) -> list[str]:
        """Return no runs: a series campaign's runs are edges, discovered during planning.

        Returns:
            An empty list.
        """
        return []

    def path_holders(self) -> list[PathHolder]:
        """Return every attribute of this request that holds an input path.

        Returns:
            Triples of holder, attribute name, and a label naming the owning entry.
        """
        holders: list[PathHolder] = [(self.protein, "path", "protein")]
        for ligand in self.ligands:
            holders.append((ligand, "structure", ligand.name))
            holders.append((ligand, "path", ligand.name))
        holders.extend((cofactor, "path", cofactor.name) for cofactor in self.cofactors)
        return holders
