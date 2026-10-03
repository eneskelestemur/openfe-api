"""The separated topologies input contract.

SepTop needs no atom mapping, so it runs the scaffold hops relative hybrid topology cannot
map. That moves the burden onto the poses: both ligands of an edge sit in the pocket at once,
each restrained to the protein, so every ligand needs a plausible bound pose in the reference
frame.
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
    "ScreeningSpec",
    "SepTopAlignmentSpec",
    "SepTopLigandSpec",
    "SepTopNetworkSpec",
    "SepTopRequest",
]


class SepTopLigandSpec(SeriesLigandSpecBase):
    """One ligand of a SepTop campaign.

    Attributes:
        pose: Overrides the campaign's ``alignment.pose`` for this ligand only.
    """

    pose: Literal["auto", "keep", "mcs", "shape"] | None = None

    @model_validator(mode="after")
    def _check_pose(self) -> SepTopLigandSpec:
        check_keep_has_coordinates(self, self.pose)
        return self


class ScreeningSpec(StrictModel):
    """Which ligands are allowed into a SepTop campaign.

    A differing net charge and a failed Boresch restraint search have no knob: the protocol
    refuses that work outright. The rest are about whether a pose is plausible, so they are
    thresholds with a policy.

    Attributes:
        policy: What a ligand failing a threshold below causes. ``warn`` keeps it and reports
            the problem; ``drop`` excludes it from the campaign.
        max_size_ratio: Largest heavy-atom count ratio between a ligand and the reference, in
            either direction. A far larger ligand is unlikely to occupy the same pocket.
        max_clashes: Heavy atoms a placed ligand may have within the clash cutoff of the
            protein. Small local clashes relax during the simulation's minimization.
        min_core_fraction: Smallest common core coverage with the reference ligand. Off by
            default, because refusing a ligand for sharing no core would refuse exactly the
            scaffold hops this protocol exists for.
    """

    policy: Literal["warn", "drop"] = "warn"
    max_size_ratio: float = Field(default=1.5, gt=1.0)
    max_clashes: int = Field(default=5, ge=0)
    min_core_fraction: float | None = Field(default=None, ge=0.0, le=1.0)


class SepTopAlignmentSpec(AlignmentSpecBase):
    """How each ligand is placed into the reference frame.

    Attributes:
        pose: Default pose source for every non-reference ligand. ``auto`` uses the supplied
            coordinates when there are any, otherwise the common core when the ligand shares
            enough of one, otherwise shape alignment onto the reference ligand. ``keep``,
            ``mcs`` and ``shape`` each force one of those paths. Whichever runs is reported
            per ligand.
        auto_core_fraction: Common core coverage at which ``auto`` places a ligand on the core
            rather than by shape.
    """

    pose: Literal["auto", "keep", "mcs", "shape"] = "auto"
    auto_core_fraction: float = Field(default=0.6, ge=0.0, le=1.0)


class SepTopNetworkSpec(StrictModel):
    """Which edges the campaign runs.

    Every pair costs the same, and no atom mapping means no per-pair difficulty, so there is
    nothing for a spanning tree to minimize. A star on the reference is then the best tree:
    every ligand sits one edge from the ligand whose pose and affinity are known.

    Attributes:
        method: How the network is built. ``radial`` connects every ligand to the hub.
            ``radial_redundant`` adds edges between consecutive spokes, closing a cycle
            through the hub so cycle closure becomes available, at roughly twice the cost.
            ``explicit`` runs exactly the edges given.
        central_ligand: Hub ligand name. Defaults to the campaign's reference ligand, and is
            rejected for ``explicit``.
        edges: Ligand name pairs, required for ``explicit`` and rejected otherwise.
    """

    method: Literal["radial", "radial_redundant", "explicit"] = "radial"
    central_ligand: str | None = Field(default=None, pattern=NAME_PATTERN)
    edges: list[tuple[str, str]] | None = None

    @model_validator(mode="after")
    def _check_method_fields(self) -> SepTopNetworkSpec:
        if self.method == "explicit" and not self.edges:
            raise ValueError(
                "'network.edges' is required for method 'explicit'; give the ligand pairs to "
                "run, for example [[lig_a, lig_b], [lig_a, lig_c]]"
            )
        if self.method != "explicit" and self.edges is not None:
            raise ValueError(
                f"'network.edges' only applies to method 'explicit', not '{self.method}'; "
                "remove it, or set 'method: explicit' to run exactly these pairs"
            )
        if self.method == "explicit" and self.central_ligand is not None:
            raise ValueError(
                "'network.central_ligand' does not apply to method 'explicit', which runs "
                "exactly the pairs in 'network.edges'; remove it"
            )
        check_edges(self.edges, self.method)
        return self


class SepTopRequest(SeriesRequestBase[SepTopLigandSpec]):
    """A complete, validated description of a SepTop campaign.

    Attributes:
        protocol: Free energy protocol to run.
        screening: Which ligands are allowed in.
        alignment: How each ligand is placed into the reference frame.
        network: Which edges to run.
    """

    protocol: Literal["septop"]
    screening: ScreeningSpec = Field(default_factory=ScreeningSpec)
    alignment: SepTopAlignmentSpec = Field(default_factory=SepTopAlignmentSpec)
    network: SepTopNetworkSpec = Field(default_factory=SepTopNetworkSpec)

    @model_validator(mode="after")
    def _check_reference_pose(self) -> SepTopRequest:
        reference = self.reference_ligand()
        if reference.pose is not None:
            raise ValueError(
                f"'pose' must not be set on the reference ligand '{reference.name}': its "
                "supplied pose is always used as given, since it defines the frame"
            )
        return self

    @model_validator(mode="after")
    def _check_network_names(self) -> SepTopRequest:
        check_network_names(self.ligand_names(), self.network.central_ligand, self.network.edges)
        return self

    def hub(self) -> str:
        """Return the ligand every other ligand connects to.

        Returns:
            ``network.central_ligand`` if set, otherwise the reference ligand's name.
        """
        return self.network.central_ligand or self.reference_ligand().name

    def pose_source(self, ligand: SepTopLigandSpec) -> Literal["auto", "keep", "mcs", "shape"]:
        """Return how one ligand's pose is obtained.

        ``auto`` is returned as such: which path it resolves to depends on the ligand's
        measured core coverage, which is only known during preparation.

        Args:
            ligand: The ligand to decide for.

        Returns:
            ``keep`` for the reference, otherwise the ligand's own ``pose`` if set, falling
            back to ``alignment.pose``.
        """
        if ligand.name == self.reference_ligand().name:
            return "keep"
        return ligand.pose or self.alignment.pose
