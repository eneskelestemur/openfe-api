"""Running repeats on the machine openfe-api is running on.

Used for a server VM or a single workstation. Tasks are spread across the available GPU
slots and run as concurrent ``openfe quickrun`` processes, with MPS packing several
repeats onto one device when the profile asks for it.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

from openfe_api.execution.base import Task, quickrun_command
from openfe_api.execution.mps import mps_daemon
from openfe_api.log import get_logger
from openfe_api.schema.profiles import ExecutionProfile

__all__ = ["LocalResult", "detect_gpus", "run_locally"]

logger = get_logger(__name__)

CommandFactory = Callable[[Task, bool], list[str]]


@dataclass
class LocalResult:
    """What happened to a batch of locally executed tasks.

    Attributes:
        completed: Labels of tasks that finished successfully.
        failed: Failure messages, keyed by task label.
        skipped: Labels of tasks that already had results.
        interrupted: Labels of tasks stopped or never started because of an interrupt.
    """

    completed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    interrupted: list[str] = field(default_factory=list)


def detect_gpus() -> list[str]:
    """Work out which GPU devices are available.

    ``CUDA_VISIBLE_DEVICES`` wins whenever it is set, since that is how a scheduler hands out
    the GPUs this job owns, and its entries pass through untouched because indices and UUIDs
    both mean something only to the driver. ``nvidia-smi`` reports every GPU on the machine
    regardless of the allocation, so it is asked only when nothing was allocated.

    Returns:
        Device identifiers, in the order they should be used.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
        if entries:
            return entries

    if shutil.which("nvidia-smi") is not None:
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            ).stdout
            indices = [line.strip() for line in output.split() if line.strip().isdigit()]
            if indices:
                return indices
        except (OSError, subprocess.SubprocessError):
            logger.debug("nvidia-smi did not report usable GPU indices")

    logger.warning("No GPU detected; running a single task at a time")
    return ["0"]


_INTERRUPT_CODES = frozenset(
    {-signal.SIGINT, -signal.SIGTERM, 128 + signal.SIGINT, 128 + signal.SIGTERM}
)
"""Exit codes of a process stopped by an interrupt or a termination request."""

_GPU_EXHAUSTION_SIGNS = (
    "no compatible cuda device",
    "out of memory",
    "cuda_error_out_of_memory",
)
"""Fragments of the errors OpenMM reports when a GPU cannot provide another context."""

_TERMINATE_GRACE = 30.0
"""Seconds a stopped process gets to exit cleanly before it is killed."""


class _RunningProcesses:
    """A thread-safe registry of the child processes currently running."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen[bytes]] = set()

    def add(self, process: subprocess.Popen[bytes]) -> None:
        """Register a started process.

        Args:
            process: The process.
        """
        with self._lock:
            self._processes.add(process)

    def discard(self, process: subprocess.Popen[bytes]) -> None:
        """Forget a finished process.

        Args:
            process: The process.
        """
        with self._lock:
            self._processes.discard(process)

    def terminate_all(self) -> None:
        """Ask every running process to stop, killing any that do not exit in time."""
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=_TERMINATE_GRACE)
            except subprocess.TimeoutExpired:
                process.kill()


def _run_one(
    task: Task,
    device: str,
    command: list[str],
    running: _RunningProcesses,
) -> bool:
    """Run one task to completion, writing its output to a log file."""
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = device

    log_path = task.work_dir / "quickrun.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, env=environment, stdout=handle, stderr=subprocess.STDOUT
        )
        running.add(process)
        try:
            returncode = process.wait()
        finally:
            running.discard(process)

    if returncode in _INTERRUPT_CODES:
        return False
    if returncode != 0:
        tail = log_path.read_text(encoding="utf-8").strip().splitlines()[-3:]
        raise RuntimeError(
            f"exit code {returncode}; see {log_path}" + (f": {' | '.join(tail)}" if tail else "")
        )
    return True


def _with_packing_hint(message: str, jobs_per_gpu: int) -> str:
    """Add context when a failure looks like the GPU refused another context.

    OpenMM reports both a full card and a card that will not accept another process as
    having no compatible device, which reads like a broken driver. Either way the first
    thing to question is how many repeats are sharing the GPU.
    """
    if jobs_per_gpu <= 1:
        return message
    if not any(sign in message.lower() for sign in _GPU_EXHAUSTION_SIGNS):
        return message
    return (
        f"{message}\nThe GPU would not give this repeat a context while {jobs_per_gpu} "
        "repeats were sharing it. Causes include the card being full, its compute mode "
        "refusing a second process, or the MPS daemon no longer running. Run "
        "'scripts/diagnose_gpu_packing.py' on the same node to tell these apart, or set "
        "'jobs_per_gpu: 1' on the execution profile to rule packing out."
    )


def run_locally(
    tasks: list[Task],
    profile: ExecutionProfile,
    force: bool = False,
    command_factory: CommandFactory = quickrun_command,
) -> LocalResult:
    """Run tasks on this machine, filling the available GPU slots.

    Tasks that already have results are skipped and tasks with a resume cache continue where
    they stopped, so calling this again after an interruption picks up the rest.

    An interrupt stops the whole batch. Stopped repeats are reported as interrupted rather
    than failed, since they can be resumed.

    Args:
        tasks: The tasks to run.
        profile: Execution profile giving the GPUs and packing to use.
        force: Whether to rerun tasks that already have results.
        command_factory: Builds the command for a task. Replaced in tests.

    Returns:
        A record of what completed, failed, was interrupted and was skipped.
    """
    pending = [task for task in tasks if force or not task.is_complete]
    result = LocalResult(skipped=[task.label for task in tasks if task not in pending])

    if not pending:
        return result

    devices = (
        [str(device) for device in profile.gpus] if profile.gpus is not None else detect_gpus()
    )
    slots: Queue[str] = Queue()
    for device in devices:
        for _ in range(profile.jobs_per_gpu):
            slots.put(device)

    logger.info(
        "Running %d task(s) across %d GPU(s) with %d job(s) per GPU",
        len(pending),
        len(devices),
        profile.jobs_per_gpu,
    )

    stop = threading.Event()
    running = _RunningProcesses()

    def worker(task: Task) -> bool:
        if stop.is_set():
            return False
        device = slots.get()
        try:
            if stop.is_set():
                return False
            task.work_dir.mkdir(parents=True, exist_ok=True)
            command = command_factory(task, task.can_resume)
            logger.info("Starting %s on GPU %s", task.label, device)
            finished = _run_one(task, device, command, running)
            if not finished:
                stop.set()
            return finished
        finally:
            slots.put(device)

    use_mps = profile.jobs_per_gpu > 1
    pipe_directory = Path(os.environ.get("TMPDIR", "/tmp")) / f"openfe-api-mps-{os.getpid()}"

    with mps_daemon(pipe_directory) if use_mps else _no_mps():
        pool = ThreadPoolExecutor(max_workers=slots.qsize())
        futures = {pool.submit(worker, task): task for task in pending}
        try:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    finished = future.result()
                except Exception as error:
                    message = _with_packing_hint(str(error), profile.jobs_per_gpu)
                    result.failed[task.label] = message
                    logger.error("Task %s failed: %s", task.label, message)
                    continue
                if finished:
                    result.completed.append(task.label)
                    logger.info("Finished %s", task.label)
        except KeyboardInterrupt:
            logger.warning("Interrupted; stopping running repeats and starting no more")
            stop.set()
            running.terminate_all()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    accounted = set(result.completed) | set(result.failed)
    result.interrupted = [task.label for task in pending if task.label not in accounted]
    if result.interrupted:
        logger.warning(
            "%d repeat(s) interrupted; run submit again to resume them",
            len(result.interrupted),
        )
    return result


class _no_mps:
    """A context manager that does nothing, used when MPS is not wanted."""

    def __enter__(self) -> bool:
        """Enter the block.

        Returns:
            False, since MPS is not running.
        """
        return False

    def __exit__(self, *_: object) -> None:
        """Leave the block."""
        return
