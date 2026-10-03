"""Preparing a series of ligands: one protein, one frame, every ligand placed into it.

RBFE and SepTop share this pipeline. The reference supplies the protein and the one pose used
as given; every other ligand passes its protocol's gate, is placed into the reference frame,
and is scored against the prepared protein.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

from openfe_api.exceptions import InputValidationError
from openfe_api.log import get_logger
from openfe_api.prep import checks
from openfe_api.prep.align import pocket_heavy_atoms, superpose
from openfe_api.prep.gate import (
    SeriesFilter,
    SeriesMember,
    clash_verdict,
    filter_series,
    screen_series,
)
from openfe_api.prep.ligand import (
    PreparedMolecule,
    build_from_file,
    build_from_smiles,
    build_molecule,
    write_sdf,
)
from openfe_api.prep.mcs import CommonCore, find_core, murcko_scaffold
from openfe_api.prep.pose import (
    clash_score,
    core_rmsd,
    generate_pose,
    shape_pose,
    transform_conformer,
)
from openfe_api.prep.protein import check_parameterizable, prepare_protein, protein_report
from openfe_api.prep.report import (
    LigandPlacement,
    MoleculeReport,
    SeriesPrepReport,
)
from openfe_api.prep.structure import Structure, StructureCache
from openfe_api.provenance import collect_provenance
from openfe_api.schema.common import Selector
from openfe_api.schema.rbfe import RbfeRequest
from openfe_api.schema.septop import SepTopRequest
from openfe_api.schema.series import SeriesLigandSpecBase

__all__ = ["SeriesRequest", "prepare_series"]

logger = get_logger(__name__)

type SeriesRequest = RbfeRequest | SepTopRequest
"""A request whose ligands are prepared together into one frame."""

PREDICTED_DISAGREEMENT = 2.0
"""Core RMSD in angstrom beyond which a ligand's own prediction is reported as disagreeing."""

MIN_CORE_FOR_PLACEMENT = 3
"""Core atoms needed to place a ligand on it: two atoms fix a position but not an orientation."""


@dataclass
class _Built:
    """A ligand as built from its inputs, before it is placed."""

    spec: SeriesLigandSpecBase
    prepared: PreparedMolecule
    source: str
    structure: Structure | None = None
    residue_index: int | None = None


def _require_selector(spec: SeriesLigandSpecBase) -> Selector:
    """Return the selector naming a ligand inside its structure file."""
    if spec.selector is None:
        raise InputValidationError(
            f"{spec.name}: a 'selector' is required to read it from {spec.structure}"
        )
    return spec.selector


def _build(spec: SeriesLigandSpecBase, cache: StructureCache, neutralize: bool) -> _Built:
    """Build one ligand from whichever inputs it supplies."""
    if spec.structure is not None:
        structure = cache.get(spec.structure)
        residue = structure.select_one_residue(_require_selector(spec), spec.name)
        prepared = build_molecule(
            spec.smiles, structure.residue_pdb_block(residue), spec.name, neutralize
        )
        return _Built(
            spec=spec,
            prepared=prepared,
            source=str(spec.structure),
            structure=structure,
            residue_index=residue.index,
        )

    if spec.path is not None:
        prepared = build_from_file(spec.smiles, spec.path, spec.name, neutralize)
        return _Built(spec=spec, prepared=prepared, source=str(spec.path))

    prepared = build_from_smiles(spec.smiles, spec.name, neutralize)
    return _Built(spec=spec, prepared=prepared, source="smiles")


def _resolve_auto(has_coordinates: bool, core: CommonCore | None, auto_core_fraction: float) -> str:
    """Choose a placement path for a ligand that asked for ``auto``.

    Supplied coordinates are trusted first, then a common core large enough to place on,
    and shape alignment covers what is left: a ligand sharing no usable core.
    """
    if has_coordinates:
        return "keep"
    if core is not None and core.size >= MIN_CORE_FOR_PLACEMENT:
        return "mcs" if core.coverage >= auto_core_fraction else "shape"
    return "shape"


def _apply_gate(
    request: SeriesRequest,
    reference_molecule: Chem.Mol,
    reference_name: str,
    candidates: dict[str, Chem.Mol],
) -> SeriesFilter:
    """Apply the gate belonging to the request's protocol."""
    if isinstance(request, RbfeRequest):
        return filter_series(reference_molecule, reference_name, candidates, request.similarity)
    return screen_series(reference_molecule, reference_name, candidates, request.screening)


def prepare_series(
    request: SeriesRequest,
    output_dir: Path,
    check_parameters: bool = True,
) -> SeriesPrepReport:
    """Prepare a series of ligands into the reference ligand's frame.

    Args:
        request: The validated RBFE or SepTop request.
        output_dir: Directory to write prepared files into. It is created if needed.
        check_parameters: Whether to check that the prepared protein parameterizes with the
            Amber force field before any planning.

    Returns:
        The preparation report, also written to ``prep_report.json``.

    Raises:
        InputValidationError: If the reference or the protein cannot be prepared, or no
            ligand passes the gate.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = StructureCache()
    warnings: list[str] = []
    excluded: set[int] = set()

    reference_spec = request.reference_ligand()
    reference = _build(reference_spec, cache, neutralize=False)
    warnings.extend(reference.prepared.warnings)
    checks.check_elements(reference.prepared.molecule, reference_spec.name)
    checks.check_radicals(reference.prepared.molecule, reference_spec.name)
    if reference.residue_index is not None:
        excluded.add(reference.residue_index)

    cofactor_reports, cofactor_excluded = _prepare_cofactors(
        request, reference, cache, output_dir, warnings
    )
    excluded |= cofactor_excluded

    protein_source = request.protein.path or reference_spec.structure
    if protein_source is None:
        raise InputValidationError(
            f"{request.name}: no source for the protein; the reference ligand "
            f"'{reference_spec.name}' must carry it, or 'protein.path' must be set"
        )

    protein_structure = cache.get(protein_source)
    protein = prepare_protein(
        protein_structure,
        output_dir / "protein.pdb",
        chains=request.protein.chains,
        ph=request.protein.ph,
        keep_waters=request.protein.keep_waters,
        exclude_residues=excluded if protein_source == reference_spec.structure else None,
    )
    warnings.extend(protein.warnings)
    if check_parameters:
        check_parameterizable(protein.path)

    reference_molecule = reference.prepared.molecule
    reference_positions = reference_molecule.GetConformer().GetPositions()
    # The prepared protein, not the input: a solvent box would clash every pose against
    # waters preparation has already dropped.
    pocket = pocket_heavy_atoms(
        Structure.load(protein.path),
        reference_positions,
        request.alignment.pocket_radius,
        exclude=set(),
    )

    built: dict[str, _Built] = {}
    for spec in request.ligands:
        if spec.name == reference_spec.name:
            continue
        try:
            built[spec.name] = _build(spec, cache, neutralize=False)
        except InputValidationError as error:
            warnings.append(f"{spec.name}: excluded, {error}")
            logger.warning("Excluded '%s': %s", spec.name, error)

    gate = _apply_gate(
        request,
        reference_molecule,
        reference_spec.name,
        {name: entry.prepared.molecule for name, entry in built.items()},
    )
    warnings.extend(gate.warnings)
    dropped = dict(gate.dropped)
    for spec in request.ligands:
        if spec.name != reference_spec.name and spec.name not in built:
            dropped[spec.name] = "could not be built from its inputs"

    placements = [_place_reference(reference, reference_molecule, pocket, output_dir, warnings)]

    requested = request.requested_poses()
    auto_core_fraction = (
        request.alignment.auto_core_fraction if isinstance(request, SepTopRequest) else 0.0
    )
    for member in gate.kept:
        entry = built[member.name]
        pose = requested[member.name]
        if pose == "auto":
            pose = _resolve_auto(entry.spec.has_coordinates(), member.core, auto_core_fraction)
        try:
            placement = _place(
                entry,
                member,
                pose,
                reference_molecule,
                request,
                cache,
                pocket,
                output_dir,
                warnings,
            )
        except InputValidationError as error:
            dropped[member.name] = str(error)
            logger.warning("Dropped '%s': %s", member.name, error)
            continue

        verdict = (
            clash_verdict(request.screening, placement.clashes)
            if isinstance(request, SepTopRequest) and request.screening.policy == "drop"
            else None
        )
        if verdict is not None:
            dropped[member.name] = f"{verdict} ('screening.policy' is 'drop')"
            logger.warning("Dropped '%s': %s", member.name, verdict)
            continue
        placements.append(placement)

    if len(placements) < 2:
        raise InputValidationError(
            f"{request.name}: only the reference ligand '{reference_spec.name}' could be "
            "placed, so there is no edge to run. The reasons the others were excluded are "
            f"in {output_dir / 'prep_report.json'}."
        )

    report = SeriesPrepReport(
        name=request.name,
        directory=output_dir,
        reference=reference_spec.name,
        protein=protein_report(protein_source, protein, check_parameters),
        ligands=placements,
        dropped=dropped,
        cofactors=cofactor_reports,
        warnings=warnings,
        provenance=collect_provenance(),
    )
    report.write(output_dir / "prep_report.json")
    logger.info(
        "Prepared series '%s': %d ligand(s) placed, %d excluded",
        request.name,
        len(placements),
        len(dropped),
    )
    return report


def _place_reference(
    reference: _Built,
    reference_molecule: Chem.Mol,
    pocket: np.ndarray,
    output_dir: Path,
    warnings: list[str],
) -> LigandPlacement:
    """Record the reference ligand, whose supplied pose defines the frame."""
    name = reference.spec.name
    core = find_core(reference_molecule, reference_molecule)
    path = output_dir / "ligands" / f"{name}.sdf"
    write_sdf(reference_molecule, path)

    clashes, contact = clash_score(reference_molecule, pocket)
    if clashes:
        warnings.append(
            f"{name}: the reference pose has {clashes} heavy atom(s) within the clash cutoff "
            f"of the prepared protein, the closest at {contact:.2f} A. Every other ligand is "
            "placed against this pose, so a poor reference carries into the whole series."
        )

    return LigandPlacement(
        name=name,
        source=reference.source,
        path=path,
        smiles=reference.prepared.smiles,
        net_charge=reference.prepared.net_charge,
        pose="keep",
        scaffold=murcko_scaffold(reference_molecule),
        core_smarts=core.smarts,
        core_size=core.size,
        core_fraction=1.0,
        perturbed_atoms=0,
        core_rmsd=0.0,
        clashes=clashes,
        worst_contact=contact,
    )


def _prepare_cofactors(
    request: SeriesRequest,
    reference: _Built,
    cache: StructureCache,
    output_dir: Path,
    warnings: list[str],
) -> tuple[list[MoleculeReport], set[int]]:
    """Build the cofactors shared by every edge, from the reference's frame."""
    reports: list[MoleculeReport] = []
    excluded: set[int] = set()

    for spec in request.cofactors:
        if spec.path is not None:
            prepared = build_from_file(spec.smiles, spec.path, spec.name)
            source: Path = spec.path
            selector = None
        else:
            reference_structure = reference.spec.structure
            if reference_structure is None or spec.selector is None:
                raise InputValidationError(
                    f"no source for cofactor '{spec.name}': set its 'path', or give the "
                    "reference ligand a combined 'structure' file and this cofactor a "
                    "'selector'"
                )
            structure = cache.get(reference_structure)
            residue = structure.select_one_residue(spec.selector, spec.name)
            prepared = build_molecule(spec.smiles, structure.residue_pdb_block(residue), spec.name)
            source = structure.path
            selector = spec.selector.describe()
            excluded.add(residue.index)

        warnings.extend(prepared.warnings)
        checks.check_elements(prepared.molecule, spec.name)
        checks.check_radicals(prepared.molecule, spec.name)
        path = output_dir / "cofactors" / f"{spec.name}.sdf"
        write_sdf(prepared.molecule, path)
        reports.append(
            MoleculeReport(
                name=spec.name,
                role="cofactor",
                source=source,
                selector=selector,
                path=path,
                smiles=prepared.smiles,
                net_charge=prepared.net_charge,
                n_atoms=prepared.molecule.GetNumAtoms(),
            )
        )

    return reports, excluded


def _place(
    entry: _Built,
    member: SeriesMember,
    pose: str,
    reference_molecule: Chem.Mol,
    request: SeriesRequest,
    cache: StructureCache,
    pocket: np.ndarray,
    output_dir: Path,
    warnings: list[str],
) -> LigandPlacement:
    """Put one gated ligand into the reference frame and score the result."""
    name = entry.spec.name
    core = member.core
    placement_rmsd: float | None = None
    superposition_rmsd: float | None = None
    predicted_rmsd: float | None = None
    shape_score: float | None = None
    shape_rmsd: float | None = None

    if pose == "keep":
        molecule = entry.prepared.molecule
        if entry.structure is not None:
            fit = superpose(
                entry.structure,
                _reference_structure(entry, request, cache),
                reference_molecule.GetConformer().GetPositions(),
                request.alignment.pocket_radius,
            )
            molecule = transform_conformer(molecule, fit.rotation, fit.translation)
            superposition_rmsd = fit.rmsd
        if core is not None:
            placement_rmsd = core_rmsd(molecule, reference_molecule, core)
            if placement_rmsd > PREDICTED_DISAGREEMENT:
                warnings.append(
                    f"{name}: its supplied pose puts the shared core {placement_rmsd:.2f} A "
                    "from the reference ligand's. Mapped atoms starting far apart converge "
                    "poorly; consider 'pose: mcs' for this ligand."
                )
    elif pose == "shape":
        overlaid = shape_pose(
            entry.prepared.molecule, reference_molecule, request.alignment, pocket=pocket
        )
        molecule = overlaid.molecule
        shape_score = overlaid.shape_score
        shape_rmsd = overlaid.shape_rmsd
        logger.info(
            "Placed '%s' by shape onto the reference (O3A score %.1f, matched RMSD %.2f A)",
            name,
            shape_score or 0.0,
            shape_rmsd or 0.0,
        )
    else:
        if core is None or core.size < MIN_CORE_FOR_PLACEMENT:
            shared = core.size if core is not None else 0
            raise InputValidationError(
                f"{name}: 'pose: mcs' places a ligand on its common core with the reference "
                f"ligand, and this ligand shares only {shared} atom(s) with it, below the "
                f"{MIN_CORE_FOR_PLACEMENT} needed to fix an orientation; use 'pose: shape' "
                "to overlay it on the reference ligand's shape instead, or supply its "
                "coordinates"
            )
        generated = generate_pose(
            entry.prepared.molecule,
            reference_molecule,
            core,
            request.alignment,
            pocket=pocket,
        )
        molecule = generated.molecule
        placement_rmsd = generated.core_rmsd
        if request.alignment.report_predicted_rmsd and entry.structure is not None:
            predicted_rmsd = _predicted_core_rmsd(entry, request, cache, reference_molecule, core)
            if predicted_rmsd is not None and predicted_rmsd > PREDICTED_DISAGREEMENT:
                warnings.append(
                    f"{name}: its own predicted pose puts the shared core "
                    f"{predicted_rmsd:.2f} A from the reference ligand's, while the "
                    "generated pose sits on it. Either the prediction disagrees with the "
                    "series' binding mode, or the common core matched the wrong atoms."
                )

    clashes, worst = clash_score(molecule, pocket)
    if clashes:
        warnings.append(
            f"{name}: {clashes} heavy atom(s) sit within the clash cutoff of the protein, "
            f"the closest at {worst:.2f} A. Small local clashes relax during the "
            "simulation's minimization; a large count means the pose does not fit."
        )

    path = output_dir / "ligands" / f"{name}.sdf"
    write_sdf(molecule, path)

    return LigandPlacement(
        name=name,
        source=entry.source,
        path=path,
        smiles=entry.prepared.smiles,
        net_charge=entry.prepared.net_charge,
        pose=pose,
        scaffold=member.scaffold,
        core_smarts=core.smarts if core is not None else None,
        core_size=core.size if core is not None else None,
        core_fraction=core.coverage if core is not None else None,
        perturbed_atoms=core.perturbed_atoms if core is not None else None,
        core_rmsd=placement_rmsd,
        shape_score=shape_score,
        shape_rmsd=shape_rmsd,
        clashes=clashes,
        worst_contact=worst,
        superposition_rmsd=superposition_rmsd,
        predicted_core_rmsd=predicted_rmsd,
    )


def _reference_structure(entry: _Built, request: SeriesRequest, cache: StructureCache) -> Structure:
    """Return the structure defining the frame, for superposing a ligand's own file onto it."""
    reference_spec = request.reference_ligand()
    source = request.protein.path or reference_spec.structure
    if source is None:
        raise InputValidationError(
            f"{entry.spec.name}: its coordinates come with their own protein, but the "
            "reference supplies no structure to superpose onto"
        )
    return cache.get(source)


def _predicted_core_rmsd(
    entry: _Built,
    request: SeriesRequest,
    cache: StructureCache,
    reference_molecule: Chem.Mol,
    core: CommonCore,
) -> float | None:
    """Measure where this ligand's own predicted pose puts the shared core.

    The coordinates are not used for the simulation; the number says whether the prediction
    agrees with the series' binding mode.
    """
    if entry.structure is None:
        return None
    try:
        fit = superpose(
            entry.structure,
            _reference_structure(entry, request, cache),
            reference_molecule.GetConformer().GetPositions(),
            request.alignment.pocket_radius,
        )
    except InputValidationError as error:
        logger.debug("Skipped the predicted pose check for '%s': %s", entry.spec.name, error)
        return None

    transplanted = transform_conformer(entry.prepared.molecule, fit.rotation, fit.translation)
    return core_rmsd(transplanted, reference_molecule, core)
