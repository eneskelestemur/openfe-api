"""CUDA MPS: sharing one GPU between concurrent repeats.

Alchemical windows rarely saturate a modern GPU on their own. Running several repeats
against one device through the Multi-Process Service raises throughput markedly, at the
cost of needing a control daemon started and stopped around the work.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from openfe_api.log import get_logger

__all__ = ["CONTROL_BINARY", "mps_daemon", "mps_shell_lines"]

logger = get_logger(__name__)

CONTROL_BINARY = "nvidia-cuda-mps-control"


@contextmanager
def mps_daemon(pipe_directory: Path) -> Iterator[bool]:
    """Run an MPS control daemon for the duration of the block.

    If the control binary is not available, the block still runs and the work simply
    proceeds without MPS.

    Args:
        pipe_directory: Directory for the daemon's pipe and log files.

    Yields:
        True if the daemon was started, False if MPS was unavailable.
    """
    if shutil.which(CONTROL_BINARY) is None:
        logger.warning("%s not found; running without CUDA MPS", CONTROL_BINARY)
        yield False
        return

    pipe_directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_directory)
    environment["CUDA_MPS_LOG_DIRECTORY"] = str(pipe_directory)

    try:
        subprocess.run([CONTROL_BINARY, "-d"], env=environment, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        logger.warning("Could not start CUDA MPS (%s); running without it", error)
        yield False
        return

    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_directory)
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(pipe_directory)
    logger.info("Started CUDA MPS control daemon in %s", pipe_directory)

    try:
        yield True
    finally:
        try:
            subprocess.run(
                [CONTROL_BINARY],
                input="quit\n",
                text=True,
                env=environment,
                check=False,
                timeout=60,
            )
            logger.info("Stopped CUDA MPS control daemon")
        except (OSError, subprocess.TimeoutExpired) as error:
            logger.warning("Could not stop CUDA MPS cleanly: %s", error)


def mps_shell_lines(pipe_directory: str) -> tuple[list[str], list[str]]:
    """Return shell lines that start and stop an MPS daemon inside a batch script.

    Args:
        pipe_directory: Shell expression for the daemon's pipe directory, normally
            containing a job-specific variable.

    Returns:
        A tuple of the lines that start the daemon and the lines that stop it.
    """
    start = [
        f'export CUDA_MPS_PIPE_DIRECTORY="{pipe_directory}"',
        f'export CUDA_MPS_LOG_DIRECTORY="{pipe_directory}"',
        'mkdir -p "$CUDA_MPS_PIPE_DIRECTORY"',
        f"if command -v {CONTROL_BINARY} >/dev/null 2>&1; then",
        f"    {CONTROL_BINARY} -d",
        "else",
        f'    echo "{CONTROL_BINARY} not found; running without MPS" >&2',
        "fi",
    ]
    stop = [
        f"if command -v {CONTROL_BINARY} >/dev/null 2>&1; then",
        f"    echo quit | {CONTROL_BINARY} || true",
        "fi",
        'rm -rf "$CUDA_MPS_PIPE_DIRECTORY"',
    ]
    return start, stop
