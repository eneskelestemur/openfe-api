"""Deciding which ligands of a series are allowed into a campaign.

Each protocol gates on what would break it. RBFE needs a shared core to mutate through, so
it gates on core coverage. SepTop needs neither a core nor a mapping, so it gates on what its
protocol refuses outright, equal net charge, and otherwise reports on how plausible a pose is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem

from openfe_api.exceptions import InputValidationError
from openfe_api.log import get_logger
from openfe_api.prep.mcs import CommonCore, find_core, murcko_scaffold, shares_chemotype
from openfe_api.schema.rbfe import SimilaritySpec
from openfe_api.schema.septop import ScreeningSpec

__all__ = [
    "SeriesFilter",
    "SeriesMember",
    "clash_verdict",
    "filter_series",
    "screen_series",
]

logger = get_logger(__name__)


@dataclass(frozen=True)
class SeriesMember:
    """A ligand that passed the gate, with what the gate measured.

    Attributes:
        name: Ligand name.
        scaffold: Murcko scaffold SMILES, used to group the series by chemotype.
        core: Its common core with the reference ligand, or None when it shares none and the
            protocol does not require one.
    """

    name: str
    scaffold: str
    core: CommonCore | None = None


@dataclass
class SeriesFilter:
    """The outcome of applying the gate to a series.

    Attributes:
        kept: Members that passed, excluding the reference.
        dropped: Why each rejected ligand was rejected, keyed by name.
        warnings: Concerns that do not reject anything.
    """

    kept: list[SeriesMember] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _reject(core: CommonCore, spec: SimilaritySpec) -> str | None:
    """Return why a common core fails the RBFE gate, or None if it passes."""
    if core.coverage < spec.min_core_fraction:
        return (
            f"shares only {core.coverage:.0%} of its heavy atoms with the reference ligand "
            f"({core.size} atoms), below the {spec.min_core_fraction:.0%} required; raise "
            "'similarity.min_core_fraction' to keep it, or run it in its own campaign"
        )
    if core.breaks_ring and not spec.allow_ring_break:
        return (
            "its common core with the reference ligand cuts through a ring, which makes the "
            "mapping unreliable; set 'similarity.allow_ring_break' to run it anyway"
        )
    if spec.max_perturbed_atoms is not None and core.perturbed_atoms > spec.max_perturbed_atoms:
        return (
            f"{core.perturbed_atoms} heavy atoms differ from the reference ligand, above the "
            f"{spec.max_perturbed_atoms} allowed by 'similarity.max_perturbed_atoms'"
        )
    return None


def filter_series(
    reference: Chem.Mol,
    reference_name: str,
    candidates: dict[str, Chem.Mol],
    spec: SimilaritySpec,
) -> SeriesFilter:
    """Apply the common core gate to every non-reference ligand of a series.

    Args:
        reference: The reference ligand.
        reference_name: Its name, for messages.
        candidates: The other ligands, keyed by name.
        spec: The gate's thresholds.

    Returns:
        The members that passed, the reasons the others did not, and any warnings.
    """
    outcome = SeriesFilter()
    reference_scaffold = murcko_scaffold(reference)

    for name, molecule in candidates.items():
        try:
            core = find_core(
                reference,
                molecule,
                core_smarts=spec.core_smarts,
                allow_ring_break=spec.allow_ring_break,
            )
        except InputValidationError as error:
            reason = (
                f"{error}, so there is nothing to mutate through; RBFE needs a congeneric "
                "series, while the SepTop protocol needs no mapping and can run this ligand"
            )
            outcome.dropped[name] = reason
            logger.warning("Dropped '%s': %s", name, reason)
            continue

        member = SeriesMember(name=name, scaffold=murcko_scaffold(molecule), core=core)
        reason = _reject(core, spec)
        if reason is not None:
            outcome.dropped[name] = reason
            logger.warning("Dropped '%s': %s", name, reason)
            continue
        outcome.kept.append(member)

    foreign = sorted(
        {
            member.scaffold
            for member in outcome.kept
            if not shares_chemotype(reference_scaffold, member.scaffold)
        }
    )
    if foreign:
        warning = (
            f"the series spans more than one chemotype: '{reference_name}' has the scaffold "
            f"{reference_scaffold or 'no ring system'}, and {len(foreign)} other scaffold(s) "
            "share no ring system with it. Relative results across chemotypes are less "
            "reliable; consider one campaign per chemotype, or a radial network with a hub "
            "per group."
        )
        outcome.warnings.append(warning)
        logger.warning(warning)

    if not outcome.kept:
        outcome.warnings.append(
            f"no ligand passed the common core gate against '{reference_name}', so there is "
            "no edge to run"
        )

    return outcome


def _size_ratio(reference: Chem.Mol, probe: Chem.Mol) -> float:
    """Return the larger of the two molecules' heavy-atom counts over the smaller."""
    counts = [
        sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1)
        for molecule in (reference, probe)
    ]
    return max(counts) / min(counts)


def clash_verdict(spec: ScreeningSpec, clashes: int) -> str | None:
    """Return why a placed pose fails the clash threshold, or None if it passes.

    The count is only known once the ligand has been placed, so this is applied after
    placement rather than in :func:`screen_series`.

    Args:
        spec: The screening thresholds.
        clashes: Heavy atoms within the clash cutoff of the protein.

    Returns:
        The failure message, or None.
    """
    if clashes <= spec.max_clashes:
        return None
    return (
        f"its placed pose has {clashes} heavy atom(s) within the clash cutoff of the "
        f"protein, above the {spec.max_clashes} allowed by 'screening.max_clashes'"
    )


def screen_series(
    reference: Chem.Mol,
    reference_name: str,
    candidates: dict[str, Chem.Mol],
    spec: ScreeningSpec,
) -> SeriesFilter:
    """Apply the SepTop screen to every non-reference ligand of a campaign.

    A net charge differing from the reference's is always rejected: the protocol itself
    refuses such a transformation, and in a star network every edge touches the reference.
    The remaining thresholds are about how plausible a pose is, so ``spec.policy`` decides
    whether failing one drops the ligand or only reports it.

    Args:
        reference: The reference ligand.
        reference_name: Its name, for messages.
        candidates: The other ligands, keyed by name.
        spec: The screen's thresholds and policy.

    Returns:
        The members that passed, the reasons the others did not, and any warnings. A member
        that shares no core with the reference carries ``core=None``, which is what makes the
        placement step fall back to shape alignment.
    """
    outcome = SeriesFilter()
    reference_charge = Chem.GetFormalCharge(reference)
    reference_scaffold = murcko_scaffold(reference)

    for name, molecule in candidates.items():
        charge = Chem.GetFormalCharge(molecule)
        if charge != reference_charge:
            reason = (
                f"its net charge of {charge:+d} differs from the reference ligand "
                f"'{reference_name}' ({reference_charge:+d}). The SepTop protocol does not "
                "support a net charge change between its end states, so no edge to the "
                "reference can be run; put this ligand in its own campaign with its own "
                "reference, or use ABFE"
            )
            outcome.dropped[name] = reason
            logger.warning("Dropped '%s': %s", name, reason)
            continue

        # Strict, as RBFE searches: a ring-breaking core maps a saturated ring onto an
        # aromatic one and scores a genuine hop at 87% shared.
        try:
            core: CommonCore | None = find_core(reference, molecule)
        except InputValidationError:
            core = None

        failures = _screen_failures(reference, molecule, core, spec)
        member = SeriesMember(name=name, scaffold=murcko_scaffold(molecule), core=core)
        if failures and spec.policy == "drop":
            reason = "; ".join(failures) + " ('screening.policy' is 'drop')"
            outcome.dropped[name] = reason
            logger.warning("Dropped '%s': %s", name, reason)
            continue

        for failure in failures:
            warning = f"{name}: {failure}, and 'screening.policy' is 'warn', so it is kept"
            outcome.warnings.append(warning)
            logger.warning(warning)
        outcome.kept.append(member)

    hops = sorted(
        {
            member.name
            for member in outcome.kept
            if member.core is None or not shares_chemotype(reference_scaffold, member.scaffold)
        }
    )
    if hops:
        outcome.warnings.append(
            f"{len(hops)} ligand(s) share no ring system with the reference "
            f"'{reference_name}': {', '.join(hops)}. SepTop runs these, which is what it is "
            "for, but their poses carry the result, so check the placement reported for each."
        )

    if not outcome.kept:
        outcome.warnings.append(
            f"no ligand passed the screen against '{reference_name}', so there is no edge to run"
        )

    return outcome


def _screen_failures(
    reference: Chem.Mol,
    molecule: Chem.Mol,
    core: CommonCore | None,
    spec: ScreeningSpec,
) -> list[str]:
    """Return every threshold in the screen that a ligand fails."""
    failures: list[str] = []

    ratio = _size_ratio(reference, molecule)
    if ratio > spec.max_size_ratio:
        failures.append(
            f"it differs from the reference ligand by a factor of {ratio:.2f} in heavy atoms, "
            f"above the {spec.max_size_ratio:.2f} allowed by 'screening.max_size_ratio'"
        )

    if spec.min_core_fraction is not None:
        coverage = core.coverage if core is not None else 0.0
        if coverage < spec.min_core_fraction:
            failures.append(
                f"it shares {coverage:.0%} of its heavy atoms with the reference ligand, "
                f"below the {spec.min_core_fraction:.0%} required by "
                "'screening.min_core_fraction'"
            )

    return failures
