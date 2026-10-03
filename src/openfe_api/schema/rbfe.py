"""The relative binding free energy input contract.

Every edge mutates one ligand into another inside a single shared protein, so all ligands must
occupy one coordinate frame. The reference entry supplies that protein and the one pose taken
as given; every other ligand is placed relative to it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from openfe_api.schema.common import NAME_PATTERN, StrictModel
from openfe_api.schema.series import (
    AlignmentSpecBase,
    SeriesLigandSpecBase,
    SeriesRequestBase,
    check_edges,
    check_keep_has_coordinates,
    check_network_names,
)

__all__ = [
    "DEFAULT_REDUNDANCY",
    "AlignmentSpec",
    "ChargeSpec",
    "NetworkSpec",
    "RbfeRequest",
    "SeriesLigandSpec",
    "SimilaritySpec",
]


class SeriesLigandSpec(SeriesLigandSpecBase):
    """One ligand of an RBFE series.

    Attributes:
        pose: Overrides the campaign's ``alignment.pose`` for this ligand only.
    """

    pose: Literal["mcs", "keep"] | None = None

    @model_validator(mode="after")
    def _check_pose(self) -> SeriesLigandSpec:
        check_keep_has_coordinates(self, self.pose)
        return self


class SimilaritySpec(StrictModel):
    """The gate deciding which ligands are similar enough to the reference to run.

    Similarity is common core coverage, not scaffold identity: a ring-size change alters a
    scaffold but is still a routine edge.

    Attributes:
        min_core_fraction: Smallest acceptable fraction of the smaller molecule's heavy atoms
            that must appear in the common core with the reference ligand.
        allow_ring_break: Whether a common core that breaks a ring is acceptable.
        max_perturbed_atoms: Largest number of heavy atoms that may differ from the
            reference. Unlimited if omitted.
        core_smarts: Explicit common core, matched against every ligand instead of searching
            for one. More reliable than a search for a series whose core is already known.
    """

    min_core_fraction: float = Field(default=0.6, ge=0.0, le=1.0)
    allow_ring_break: bool = False
    max_perturbed_atoms: int | None = Field(default=None, ge=1)
    core_smarts: str | None = Field(default=None, min_length=1)


class AlignmentSpec(AlignmentSpecBase):
    """How each ligand is placed into the reference frame.

    Attributes:
        pose: Default pose source for every non-reference ligand. ``mcs`` generates conformers
            restrained to the reference ligand's common core, keeping mapped atoms in
            consistent geometry across the network. ``keep`` uses the supplied coordinates, as
            a docking run into a fixed receptor already provides; a kept file carrying its own
            protein is superposed onto the reference first.
    """

    pose: Literal["mcs", "keep"] = "mcs"


DEFAULT_REDUNDANCY = 2
"""Independent paths per ligand for a redundant network."""


class NetworkSpec(StrictModel):
    """Which edges the campaign runs.

    Attributes:
        method: How the network is built. ``minimal_redundant`` adds cycles, which cost more
            edges but survive a failed run and give cycle closure as a quality check.
            ``minimal_spanning`` is the cheapest connected network. ``radial`` connects every
            ligand to one hub. ``explicit`` runs exactly the edges given.
        redundancy: Number of independent paths per ligand, for ``minimal_redundant``.
        mapper: Atom mapper to use. ``lomap`` maps on 2D topology and is insensitive to pose
            quality; ``kartograf`` maps on 3D geometry; ``both`` keeps the better scoring one.
        central_ligand: Hub ligand name, required for ``radial`` and rejected otherwise.
        edges: Ligand name pairs, required for ``explicit`` and rejected otherwise.
    """

    method: Literal["minimal_redundant", "minimal_spanning", "radial", "explicit"] = (
        "minimal_redundant"
    )
    redundancy: int = Field(default=DEFAULT_REDUNDANCY, ge=2)
    mapper: Literal["lomap", "kartograf", "both"] = "lomap"
    central_ligand: str | None = Field(default=None, pattern=NAME_PATTERN)
    edges: list[tuple[str, str]] | None = None

    @model_validator(mode="after")
    def _check_method_fields(self) -> NetworkSpec:
        if self.method == "explicit" and not self.edges:
            raise ValueError(
                "'network.edges' is required for method 'explicit'; give the ligand pairs to "
                "run, for example [[lig_a, lig_b], [lig_b, lig_c]]"
            )
        if self.method != "explicit" and self.edges is not None:
            raise ValueError(
                f"'network.edges' only applies to method 'explicit', not '{self.method}'; "
                "remove it, or set 'method: explicit' to run exactly these pairs"
            )
        if self.method == "radial" and self.central_ligand is None:
            raise ValueError(
                "'network.central_ligand' is required for method 'radial'; name the ligand "
                "every other ligand connects to"
            )
        if self.method != "radial" and self.central_ligand is not None:
            raise ValueError(
                f"'network.central_ligand' only applies to method 'radial', not "
                f"'{self.method}'; remove it"
            )
        # Against the default, not model_fields_set: a manifest serializes every field, so a
        # set-ness check would refuse to reload a campaign it just wrote.
        if self.method != "minimal_redundant" and self.redundancy != DEFAULT_REDUNDANCY:
            raise ValueError(
                f"'network.redundancy' only applies to method 'minimal_redundant', not "
                f"'{self.method}'; remove it"
            )
        check_edges(self.edges, self.method)
        return self


class ChargeSpec(StrictModel):
    """What to do with edges that change the ligand's net charge.

    Attributes:
        correct_single: Whether a charge difference of exactly one is corrected by transforming
            a water into a counterion. On by default, and roughly four times the cost of a
            neutral edge: 22 lambda windows sampled for 20 ns each.
        allow_multi: Whether edges changing the net charge by more than one may run. OpenFE has
            no correction for these, so the planner routes around them by default.
    """

    correct_single: bool = True
    allow_multi: bool = False


class RbfeRequest(SeriesRequestBase[SeriesLigandSpec]):
    """A complete, validated description of an RBFE campaign.

    Attributes:
        protocol: Free energy protocol to run.
        similarity: The common core gate applied before any alignment.
        alignment: How each ligand is placed into the reference frame.
        network: Which edges to run.
        charges: How charge-changing edges are handled.
    """

    protocol: Literal["rbfe"]
    similarity: SimilaritySpec = Field(default_factory=SimilaritySpec)
    alignment: AlignmentSpec = Field(default_factory=AlignmentSpec)
    network: NetworkSpec = Field(default_factory=NetworkSpec)
    charges: ChargeSpec = Field(default_factory=ChargeSpec)

    @model_validator(mode="after")
    def _check_reference_pose(self) -> RbfeRequest:
        reference = self.reference_ligand()
        if reference.pose is not None:
            raise ValueError(
                f"'pose' must not be set on the reference ligand '{reference.name}': its "
                "supplied pose is always used as given, since it defines the frame"
            )
        return self

    @model_validator(mode="after")
    def _check_network_names(self) -> RbfeRequest:
        check_network_names(self.ligand_names(), self.network.central_ligand, self.network.edges)
        return self

    def pose_source(self, ligand: SeriesLigandSpec) -> Literal["mcs", "keep"]:
        """Return how one ligand's pose is obtained.

        Args:
            ligand: The ligand to decide for.

        Returns:
            ``keep`` for the reference, otherwise the ligand's own ``pose`` if set, falling
            back to ``alignment.pose``.
        """
        if ligand.name == self.reference_ligand().name:
            return "keep"
        return ligand.pose or self.alignment.pose
