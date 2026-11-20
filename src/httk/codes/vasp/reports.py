"""Supervised VASP execution and its classified run report.

This is the execution side of the dependency-free VASP interface: a live monitor
that turns VASP-5/6 output lines into diagnostics, :func:`run_vasp` which drives
one supervised process and combines the live and file-level diagnostics into a
classified :class:`VaspRunReport`. Historical authorship is documented in
``v1_runtime/NOTICE``.
"""

import logging
import math
import os
import re
import shlex
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from httk.workflow.codes import (
    CheckerSpec,
    Diagnostic,
    FollowSource,
    ProcessReport,
    ProcessSupervisor,
    SourceEvent,
    launch_command,
    read_json,
    write_json_atomic,
)

from .diagnostics import (
    _ALLOCATION_CODE,
    _ELECTRONIC_STEP,
    _IONIC_STEP,
    _NSW,
    VASP_CHECKS,
    VaspCheck,
    diagnose_vasp_files,
    parse_diagnostic_policy,
)
from .inputs import _write_text_atomic

_LOGGER = logging.getLogger(__name__)

#: The event sources of VASP's own output: the two followed files and the process streams.
_VASP_SOURCES = frozenset({"OUTCAR", "OSZICAR", "stdout", "stderr"})
#: The only files a checker may follow inside the run directory: each is removed before the run.
_FOLLOWABLE_OUTPUTS = (
    "OUTCAR",
    "OSZICAR",
    "vasp.out",
    "vasp.err",
    "REPORT",
    "XDATCAR",
    "PCDAT",
    "EIGENVAL",
    "DOSCAR",
    "PROCAR",
    "IBZKPT",
    "vasprun.xml",
    "ML_LOGFILE",
)
#: The STOPCAR each ``soft_stop`` mode writes.
_STOPCARS = {"ionic": "LSTOP = .TRUE.\n", "electronic": "LABORT = .TRUE.\n"}
#: How much of the end of OUTCAR and ``vasp.out`` is searched for VASP's stop and convergence messages.
_TAIL_BYTES = 1 << 20


def _checker_source_problem(path: Path) -> str | None:
    """Return why a checker may not follow *path*, relative to the run directory, or ``None``.

    Lexical on purpose: a symlink is never resolved, so it cannot redirect the pre-run removal.
    """

    relative = os.path.normpath(path)
    if os.path.isabs(relative) or relative == os.pardir or relative.startswith(os.pardir + os.sep):
        return None
    if relative in _FOLLOWABLE_OUTPUTS:
        return None
    return (
        f"checker source {os.fspath(path)!r}: a checker inside the run directory may only follow "
        f"VASP's own output files ({', '.join(_FOLLOWABLE_OUTPUTS)})"
    )


class _VaspMonitor:
    def __init__(self, checks: Sequence[VaspCheck], policy: Mapping[str, str]) -> None:
        self.checks = tuple(checks)
        self.policy = policy

    def __call__(self, event: SourceEvent) -> Sequence[Diagnostic]:
        # Checker sources are followed too; the built-in checks read only VASP's own output.
        if event.event != "line" or event.line is None or event.source not in _VASP_SOURCES:
            return ()
        result: list[Diagnostic] = []
        allocation = re.search(r"total allocation\s*:\s*([0-9]+)\s*KBytes", event.line, re.IGNORECASE)
        if allocation is not None and int(allocation.group(1)) > 500_000:
            result.append(
                Diagnostic(
                    _ALLOCATION_CODE,
                    "fatal",
                    "VASP requested more than 500000 KBytes for real-space projection",
                    event.source,
                    event.line,
                    True,
                )
            )
        for check in self.checks:
            if check.pattern.search(event.line):
                result.append(
                    Diagnostic(check.code, check.severity, event.line.strip(), event.source, event.line, check.stop)
                )
        return [
            item if action is None else replace(item, stop=action == "stop")
            for item in result
            if (action := self.policy.get(item.code)) != "ignore"
        ]


class _SoftStop:
    """Write the STOPCAR of *mode* once, when the clock reaches the stop time.

    The stop time is the earliest of *stop_warning* and, once a step has completed,
    *deadline* less three average steps (custodian's ``WalltimeHandler`` rule).
    """

    def __init__(
        self,
        mode: str,
        directory: Path,
        stop_warning: float | None,
        deadline: float | None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = directory / "STOPCAR"
        self.content = _STOPCARS[mode]
        self.step = _IONIC_STEP if mode == "ionic" else _ELECTRONIC_STEP
        self.stop_warning = stop_warning
        self.deadline = deadline
        self.clock = clock
        self.started: float | None = None
        self.steps = 0
        self.last_step = 0.0
        self.requested_at: float | None = None
        self.done = False

    def __call__(self, event: SourceEvent) -> Sequence[Diagnostic]:
        now = self.clock()
        if self.started is None:
            self.started = now
        if self.done:
            return ()
        if event.event == "line" and event.source == "OSZICAR" and event.line and self.step.match(event.line):
            self.steps += 1
            self.last_step = now
        times = [] if self.stop_warning is None else [self.stop_warning]
        if self.deadline is not None and self.steps:
            times.append(self.deadline - 3 * (self.last_step - self.started) / self.steps)
        if times and now >= min(times):
            self.done = True
            try:
                _write_text_atomic(self.path, self.content)
            except OSError as exc:
                # A monitor exception would stop VASP as monitor_error: the run outlives a failed soft stop.
                _LOGGER.warning("could not write %s, VASP runs on to the deadline: %s", self.path, exc)
            else:
                self.requested_at = now
        return ()


def _environment_time(value: float | None, variable: str) -> float | None:
    """Return *value*, else the epoch second in the environment *variable*, else ``None``."""

    if value is not None:
        return value
    try:
        found = float(os.environ[variable])
    except (KeyError, ValueError):
        return None
    return found if math.isfinite(found) else None


def _tail(path: Path) -> str:
    """Return the lowercased last :data:`_TAIL_BYTES` of *path*, or ``""`` when it is not a file."""

    # ponytail: a bounded tail, since an OUTCAR can be hundreds of MB; VASP prints these messages
    # just before the footer, so only more than 1 MiB of output after them would hide them.
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, stream.seek(0, os.SEEK_END) - _TAIL_BYTES))
        return stream.read().decode("utf-8", errors="replace").lower()


def _stop_acknowledgement(root: Path) -> str | None:
    """Return ``"hard"`` or ``"soft"`` when VASP's OUTCAR or captured stdout acknowledges a STOPCAR."""

    tails = (_tail(root / "OUTCAR"), _tail(root / "vasp.out"))
    if any("hard stop encountered" in text for text in tails):
        return "hard"
    if any("soft stop encountered" in text for text in tails):
        return "soft"
    return None


def _soft_stop_ended_early(root: Path) -> bool:
    """Return whether a soft-stopped relaxation ended before ``NSW`` ionic steps without converging."""

    outcar, oszicar = root / "OUTCAR", root / "OSZICAR"
    nsw = None
    if outcar.is_file():
        # Streamed to the header's first NSW echo, whatever precedes it.
        with outcar.open(encoding="utf-8", errors="replace") as stream:
            nsw = next((int(match.group(1)) for line in stream if (match := _NSW.match(line))), None)
    if nsw is None or nsw <= 1 or "reached required accuracy" in _tail(outcar):
        return False
    lines = oszicar.read_text(encoding="utf-8", errors="replace").splitlines() if oszicar.is_file() else []
    return sum(1 for line in lines if _IONIC_STEP.match(line)) < nsw


@dataclass(frozen=True)
class VaspRunReport:
    """Classified result of one supervised VASP execution.

    :param process: Preserve the supervised process result.
    :param classification: Record the final VASP run classification.
    :param diagnostics: Preserve live and file-level diagnostics.
    :param diagnostic_policy: Record the policy applied to the live checks, by diagnostic code.
    :param soft_stop: Record the soft stop: its ``mode`` (``"ionic"``, ``"electronic"`` or ``None``),
        ``requested_at`` (the epoch second the STOPCAR was written, else ``None``), ``lead_seconds``
        (the deadline less ``requested_at`` when both are known, else ``None``) and ``acknowledged``
        (``"soft"`` or ``"hard"`` when VASP's output reports a STOPCAR stop, else ``None``).
    """

    process: ProcessReport
    classification: str
    diagnostics: tuple[Diagnostic, ...]
    diagnostic_policy: Mapping[str, str] = field(default_factory=dict)
    soft_stop: Mapping[str, object] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, object]:
        """Serialize the report for JSON storage.

        :return: The JSON-compatible report mapping.
        """
        return {
            "format": "httk-vasp-run-report",
            "format_version": 2,
            "process": self.process.as_mapping(),
            "classification": self.classification,
            "diagnostics": [item.as_mapping() for item in self.diagnostics],
            "diagnostic_policy": dict(self.diagnostic_policy),
            "soft_stop": dict(self.soft_stop),
        }

    def write(self, path: str | os.PathLike[str]) -> Path:
        """Write the report as JSON.

        :param path: Write the report to this path.
        :return: The report path.
        """
        destination = Path(path)
        write_json_atomic(destination, self.as_mapping())
        return destination


def run_vasp(
    argv: Sequence[str],
    *,
    directory: str | os.PathLike[str] = ".",
    timeout: float | None = None,
    launch: bool | None = None,
    termination_grace: float = 10.0,
    report_path: str | os.PathLike[str] = "vasp-run-report.json",
    dispatch: bool = True,
    checks: Sequence[VaspCheck] = VASP_CHECKS,
    policy: Mapping[str, str] | str | None = None,
    monitors: Sequence[Callable[[SourceEvent], Sequence[Diagnostic]]] = (),
    checkers: Sequence[CheckerSpec] = (),
    soft_stop: str | None = None,
    stop_warning: float | None = None,
    deadline: float | None = None,
) -> VaspRunReport:
    """Run VASP with live VASP-5/6 diagnostics and a structured report.

    By default VASP starts through the launch wrapper :mod:`httk.codes.vasp.launch`:
    *argv* is its standard command, the ``vasp.ncl_command`` and
    ``vasp.gamma_command`` settings (job parameter, ``HTTK_`` variable or
    workspace setting) its non-collinear and Gamma-only commands (*argv* again for
    a non-collinear run when ``vasp.ncl_command`` is unset), and the wrapper
    writes the planned ``KPAR``/``NCORE`` into ``INCAR`` and applies the launch
    prefix. The followed ``OUTCAR`` and ``OSZICAR`` are removed before VASP
    starts, so a stale file from an earlier run cannot replay its diagnostics into
    this one; VASP rewrites both anyway. A ``STOPCAR`` is removed before VASP starts
    and again after it ends, whatever *soft_stop*, since a stale one would stop the
    next run at once. ``CONTCAR`` and the run report are not touched.

    Live supervision applies *checks* to every followed line, adjusted by
    *policy*, then the extra *monitors* and *checkers*; any diagnostic with
    ``stop`` set ends the run as ``diagnosed_stop``. A monitor or checker that
    emits a built-in code gets that code's remedy from the remedy planner.

    With *soft_stop* set, a monitor writes the mode's ``STOPCAR`` once, when the
    clock reaches the earliest of *stop_warning* and, once a step has completed,
    *deadline* less three average steps (ionic steps for ``"ionic"``, electronic
    steps for ``"electronic"``); with neither time known it never writes one.
    VASP then ends cleanly with exit status 0. A run whose ``OUTCAR`` or ``vasp.out``
    acknowledges a STOPCAR, whoever wrote it, is classified ``stopped`` instead
    of ``nonconverged`` or a missing-footer ``process_failure``: always for
    ``hard stop encountered`` (``LABORT``), and for ``soft stop encountered``
    (``LSTOP``) only when ``NSW`` exceeds 1, fewer than ``NSW`` ionic steps
    completed and ``reached required accuracy`` is absent. A timeout, a stop
    diagnostic or a nonzero exit status keeps its own classification.

    :param argv: Execute this VASP command argument vector, naming the program (for example ``["vasp_std"]``).
        The attempt's launch prefix (``HTTK_WORKFLOW_LAUNCH``) is prepended by default; a command that already
        starts with a launcher such as ``srun`` or ``mpirun`` is refused with :class:`ValueError` when a prefix applies.
    :param directory: Run VASP in this directory.
    :param timeout: Stop the process after this duration when set.
    :param launch: Prepend the attempt's launch prefix when true, the default (``None``);
        ``False`` runs *argv* as given.
    :param termination_grace: Allow this duration for graceful termination.
    :param report_path: Write the structured report at this workdir-relative path.
    :param dispatch: Start VASP through the launch wrapper, the default; ``False`` runs *argv*
        (behind the launch prefix) without binary selection or parallelization planning.
    :param checks: Apply these live checks, by default :data:`~httk.codes.vasp.VASP_CHECKS`.
    :param policy: Override the built-in checks per code (see :func:`~httk.codes.vasp.parse_diagnostic_policy`):
        ``stop`` ends the run, ``continue`` reports without stopping, ``ignore`` drops the diagnostic.
        It governs the built-in live checks only, not *monitors*, *checkers* or the post-run file diagnosis.
    :param monitors: Also call these in-process monitors on every supervision event.
    :param checkers: Also run these executable checkers; their ``sources`` are followed too, a relative
        path resolved against *directory*. Inside the run directory only VASP's own output files
        (``OUTCAR``, ``OSZICAR``, ``vasp.out``, ``vasp.err``, ``REPORT``, ``XDATCAR``, ``PCDAT``,
        ``EIGENVAL``, ``DOSCAR``, ``PROCAR``, ``IBZKPT``, ``vasprun.xml``, ``ML_LOGFILE``) may be followed,
        and they are removed before each run; files outside it are followed and never touched. A checker's
        ``argv`` runs unchanged from the caller's working directory (the job workdir for a workflow runner),
        so name it by an absolute path otherwise.
    :param soft_stop: Write a STOPCAR before the deadline: ``"ionic"`` writes ``LSTOP = .TRUE.`` (VASP stops
        after the current ionic step, for a relaxation that resumes from ``CONTCAR``), ``"electronic"``
        writes ``LABORT = .TRUE.`` (VASP stops after the current electronic step); ``None``, the default,
        never writes one.
    :param stop_warning: Write the STOPCAR from this epoch second at the latest; ``None`` reads
        ``HTTK_WORKFLOW_STOP_WARNING``. Used only with *soft_stop*.
    :param deadline: The epoch second the attempt is stopped at; ``None`` reads ``HTTK_WORKFLOW_DEADLINE``.
        Used only with *soft_stop*.
    :return: The classified VASP run report; its ``process.argv`` is the wrapper command line when dispatching.
    :raises ValueError: If a command, or a configured ``vasp.ncl_command``/``vasp.gamma_command``,
        starts with a launcher while a launch prefix applies, if *policy* is invalid, or if a checker
        follows another file inside the run directory, or one there that is a symlink or not a regular file,
        if *soft_stop* is not ``None``, ``"ionic"`` or ``"electronic"``, or if ``STOPCAR`` is a symlink or
        not a regular file.
    """

    # Parsed first: an invalid policy must not touch the workdir.
    diagnostic_policy = parse_diagnostic_policy(policy, checks=checks)
    if soft_stop is not None and soft_stop not in _STOPCARS:
        raise ValueError(f"soft_stop must be None, 'ionic' or 'electronic', not {soft_stop!r}")
    root = Path(directory).resolve()
    stopcar = root / "STOPCAR"
    follow = [FollowSource(root / "OSZICAR", "OSZICAR"), FollowSource(root / "OUTCAR", "OUTCAR")]
    # Everything is checked before anything is removed, lexically: a symlink is never followed for deletion.
    for checker in checkers:
        for source in checker.sources:
            problem = _checker_source_problem(Path(os.path.relpath(root / source.path, root)))
            if problem is not None:
                raise ValueError(problem)
            path = Path(os.path.normpath(root / source.path))
            if path.is_relative_to(root) and (path.is_symlink() or (path.exists() and not path.is_file())):
                raise ValueError(f"checker source {path} is not a regular file; it would be removed before the run")
            if path not in {item.path for item in follow}:
                follow.append(replace(source, path=path))
    if stopcar.is_symlink() or (stopcar.exists() and not stopcar.is_file()):
        raise ValueError(f"{stopcar} is not a regular file; it would be removed before the run")
    # A stale source would replay the previous run into the checks; one outside the run directory is not ours.
    for source in follow:
        if source.path.is_relative_to(root):
            source.path.unlink(missing_ok=True)
    stopcar.unlink(missing_ok=True)
    # Refuses a leading launcher before anything starts, with or without the wrapper.
    command = launch_command(argv, launch=launch is not False)
    if dispatch:
        # Imported here: the package importing launch would make ``python -m httk.codes.vasp.launch`` warn.
        from .launch import _setting

        variants = {
            name: str(value) for name in ("vasp.ncl_command", "vasp.gamma_command") if (value := _setting(name))
        }
        for name, value in variants.items():
            try:
                launch_command(shlex.split(value), launch=launch is not False)
            except ValueError as exc:
                raise ValueError(f"the {name} setting: {exc}") from exc
        standard = shlex.join(argv)
        # Without vasp.ncl_command, a non-collinear INCAR runs argv as it did before the wrapper.
        command = [
            sys.executable,
            "-P",
            "-m",
            "httk.codes.vasp.launch",
            f"--standard={standard}",
            f"--ncl={variants.get('vasp.ncl_command', standard)}",
        ]
        if "vasp.gamma_command" in variants:
            command.append(f"--gamma={variants['vasp.gamma_command']}")
        if launch is False:
            command.append("--no-launch")
    stopper = None
    if soft_stop is not None:
        deadline = _environment_time(deadline, "HTTK_WORKFLOW_DEADLINE")
        stopper = _SoftStop(soft_stop, root, _environment_time(stop_warning, "HTTK_WORKFLOW_STOP_WARNING"), deadline)
    supervisor = ProcessSupervisor(
        monitors=(_VaspMonitor(checks, diagnostic_policy), *monitors, *(() if stopper is None else (stopper,))),
        checkers=checkers,
        follow=follow,
    )
    process = supervisor.run(
        command,
        timeout=timeout,
        cwd=root,
        termination_grace=termination_grace,
        stdout_path=root / "vasp.out",
        stderr_path=root / "vasp.err",
    )
    stopcar.unlink(missing_ok=True)
    diagnostics = _deduplicate((*process.diagnostics, *diagnose_vasp_files(root)))
    acknowledged = _stop_acknowledgement(root)
    if process.timed_out:
        classification = "timeout"
    elif any(item.stop for item in diagnostics):
        classification = "diagnosed_stop"
    elif process.returncode:
        classification = "process_failure"
    elif acknowledged == "hard" or (acknowledged == "soft" and _soft_stop_ended_early(root)):
        classification = "stopped"
    elif any(
        item.code in {"electronic_nonconvergence", "ionic_nonconvergence", "positive_final_energy"}
        for item in diagnostics
    ):
        classification = "nonconverged"
    elif any(item.code == "incomplete_outcar" for item in diagnostics):
        classification = "process_failure"
    else:
        classification = "completed"
    requested_at = None if stopper is None else stopper.requested_at
    lead = None if requested_at is None or deadline is None else deadline - requested_at
    record = {"mode": soft_stop, "requested_at": requested_at, "lead_seconds": lead, "acknowledged": acknowledged}
    report = VaspRunReport(process, classification, diagnostics, diagnostic_policy, record)
    report.write(root / report_path)
    return report


def soft_stop_mintime(report: VaspRunReport) -> str | None:
    """Return the ``mintime`` to resume a soft-stopped run with.

    It is twice the report's ``lead_seconds``, rounded up to whole seconds and at
    least one minute, so a manager about to drain does not start the resumed run.

    :param report: Read this run report.
    :return: The Slurm duration ``H:MM:SS``, or ``None`` when the run was not ``stopped``
        or its lead is unknown.
    """

    return _mintime(report.classification, report.soft_stop)


def _mintime(classification: object, soft_stop: object) -> str | None:
    """Return :func:`soft_stop_mintime` of a classification and a ``soft_stop`` member read from JSON."""

    lead = soft_stop.get("lead_seconds") if isinstance(soft_stop, Mapping) else None
    if classification != "stopped" or not isinstance(lead, int | float) or isinstance(lead, bool):
        return None
    seconds = max(60, math.ceil(2 * lead))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def read_checker_specs(value: object, *, base: str | os.PathLike[str]) -> tuple[CheckerSpec, ...]:
    """Read executable checker specifications shipped as files.

    Each named file is an ``httk-workflow-checker-spec`` version 2 JSON object.
    So that a checker shipped beside its spec works from any workdir, an ``argv``
    element starting with ``./`` names a file in the spec file's directory and
    becomes that file's absolute path; other elements (``awk``, ``-f``, flags,
    ``POSCAR``) are kept as written.

    :param value: Read the spec files named by this space-separated string or list
        of strings, or ``None``/``""`` for none.
    :param base: Resolve relative spec paths against this directory, usually the job payload.
    :return: The checker specifications, in the order named.
    :raises ValueError: If *value* is malformed, a spec file is missing, unreadable or invalid,
        or a ``./`` argument names no file beside its spec.
    """

    if value is None or value == "":
        return ()
    if isinstance(value, str):
        names: list[object] = list(value.split())
    elif isinstance(value, Sequence) and not isinstance(value, bytes):
        names = list(value)
    else:
        raise ValueError(
            f"checkers must be a list or a space-separated string of spec paths, not {type(value).__name__}"
        )
    specs: list[CheckerSpec] = []
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"checkers entry {name!r} is not a nonempty spec path")
        path = Path(base, name).resolve()
        try:
            spec = CheckerSpec.from_mapping(read_json(path))
        except ValueError as exc:  # read_json's FormatError covers OSError and JSON errors
            raise ValueError(f"checker spec {path}: {exc}") from exc
        # Absolute sources are left to run_vasp, which knows the run directory.
        for source in spec.sources:
            if not source.path.is_absolute() and (problem := _checker_source_problem(source.path)) is not None:
                raise ValueError(f"checker spec {path}: {problem}")
        argv: list[str] = []
        for item in spec.argv:
            if item.startswith("./"):
                shipped = (path.parent / item).resolve()
                if not shipped.is_file():
                    raise ValueError(f"checker spec {path}: argument {item!r} names no file beside the spec")
                item = str(shipped)
            argv.append(item)
        specs.append(replace(spec, argv=tuple(argv)))
    return tuple(specs)


def _deduplicate(values: Sequence[Diagnostic]) -> tuple[Diagnostic, ...]:
    result: list[Diagnostic] = []
    seen: set[tuple[str, str, str | None]] = set()
    for item in values:
        key = item.code, item.source, item.evidence
        if key not in seen:
            seen.add(key)
            result.append(item)
    return tuple(result)
