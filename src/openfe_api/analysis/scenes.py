"""Writing a pose out as something you can look at.

Nothing here imports PyMOL: a scene is a script PyMOL opens. The same frame is also written
as a self-contained page, so a client with no molecular viewer still gets a look at the pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from string import Template

import MDAnalysis as mda
import numpy as np
import py3Dmol

from openfe_api.analysis.interactions import POCKET_RADIUS
from openfe_api.analysis.trajectory import (
    LIGAND_RESNAME,
    LIGAND_SELECTION,
    TrajectorySource,
    open_trajectory,
)
from openfe_api.exceptions import InputValidationError
from openfe_api.log import get_logger

__all__ = ["SCENES_DIR", "Scene", "representative_frame", "write_scene"]

logger = get_logger(__name__)

SCENES_DIR = "scenes"
TEMPLATE = "scene.pml"
VIEWER_SIZE = (720, 540)


@dataclass(frozen=True)
class Scene:
    """Where one run's pose was written.

    Attributes:
        run: Run name.
        frame: Which frame of the trajectory it is.
        structure: The frame as a PDB.
        script: A PyMOL script that opens it.
        page: A self-contained page showing the same pose.
    """

    run: str
    frame: int
    structure: Path
    script: Path
    page: Path


def representative_frame(universe: mda.Universe, ligand: mda.AtomGroup) -> int:
    """Return the frame whose ligand sits closest to the run's median position.

    The median rather than the mean, so one frame where the ligand swung away does not drag
    the choice with it.

    Args:
        universe: The open trajectory.
        ligand: The ligand to follow.

    Returns:
        The index of the chosen frame.
    """
    centers = np.array([ligand.center_of_mass() for _ in universe.trajectory])
    median = np.median(centers, axis=0)
    return int(np.linalg.norm(centers - median, axis=1).argmin())


def write_scene(source: TrajectorySource, run: str, directory: Path) -> Scene:
    """Write one run's representative pose as a structure, a script and a page.

    Args:
        source: The trajectory and its topology.
        run: Run name, used for the file names.
        directory: Where to write the scene.

    Returns:
        What was written.

    Raises:
        InputValidationError: If the trajectory holds no ligand to center the scene on.
    """
    universe = open_trajectory(source)
    try:
        ligand = universe.select_atoms(LIGAND_SELECTION)
        if not len(ligand):
            raise InputValidationError(f"{source.trajectory}: no ligand to build a scene around.")

        frame = representative_frame(universe, ligand)
        universe.trajectory[frame]
        # The whole protein, not just the pocket: the scene scripts select the pocket
        # themselves, and a cartoon of loose residues reads as nothing at all.
        keep = universe.select_atoms(f"protein or {LIGAND_SELECTION}")

        directory.mkdir(parents=True, exist_ok=True)
        structure = directory / f"{run}.pdb"
        with mda.Writer(str(structure), n_atoms=len(keep)) as writer:
            writer.write(keep)

        script = _write_script(structure, run, directory)
        page = _write_page(structure, run, directory)
    finally:
        universe.trajectory.close()

    logger.info("Wrote the scene for '%s' from frame %d", run, frame)
    return Scene(run=run, frame=frame, structure=structure, script=script, page=page)


def _write_script(structure: Path, run: str, directory: Path) -> Path:
    """Fill the PyMOL template for one structure."""
    source = resources.files("openfe_api.analysis.templates").joinpath(TEMPLATE)
    filled = Template(source.read_text(encoding="utf-8")).substitute(
        title=run, structure=structure.name, radius=f"{POCKET_RADIUS:.0f}"
    )
    path = directory / f"{run}.pml"
    path.write_text(filled, encoding="utf-8")
    return path


def _write_page(structure: Path, run: str, directory: Path) -> Path:
    """Write the same pose as a page that opens in a browser."""
    viewer = py3Dmol.view(width=VIEWER_SIZE[0], height=VIEWER_SIZE[1])
    viewer.addModel(structure.read_text(encoding="utf-8"), "pdb")
    viewer.setStyle({"cartoon": {"color": "spectrum", "opacity": 0.55}})
    ligand = {"resn": LIGAND_RESNAME}
    viewer.addStyle(ligand, {"stick": {"radius": 0.18, "colorscheme": "cyanCarbon"}})
    viewer.addStyle(
        {"byres": True, "within": {"distance": POCKET_RADIUS, "sel": ligand}},
        {"stick": {"radius": 0.08, "colorscheme": "whiteCarbon"}},
    )
    viewer.zoomTo(ligand)
    path = directory / f"{run}.html"
    viewer.write_html(str(path), fullpage=False)
    return path
