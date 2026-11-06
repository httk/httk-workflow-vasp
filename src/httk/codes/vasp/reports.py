"""Supervised VASP execution and its classified run report.

This is the execution side of the dependency-free VASP interface: a live monitor
that turns VASP-5/6 output lines into diagnostics, :func:`run_vasp` which drives
one supervised process and combines the live and file-level diagnostics into a
classified :class:`VaspRunReport`. Historical authorship is documented in
``v1_runtime/NOTICE``.
"""

import os
import re
import shlex
import sys
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

from .diagnostics import _ALLOCATION_CODE, VASP_CHECKS, VaspCheck, diagnose_vasp_files, parse_diagnostic_policy

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


@dataclass(frozen=True)
class VaspRunReport:
    """Classified result of one supervised VASP execution.

    :param process: Preserve the supervised process result.
    :param classification: Record the final VASP run classification.
    :param diagnostics: Preserve live and file-level diagnostics.
    :param diagnostic_policy: Record the policy applied to the live checks, by diagnostic code.
    """

    process: ProcessReport
    classification: str
    diagnostics: tuple[Diagnostic, ...]
    diagnostic_policy: Mapping[str, str] = field(default_factory=dict)

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
    this one; VASP rewrites both anyway. ``CONTCAR`` and the run report are not touched.

    Live supervision applies *checks* to every followed line, adjusted by
    *policy*, then the extra *monitors* and *checkers*; any diagnostic with
    ``stop`` set ends the run as ``diagnosed_stop``. A monitor or checker that
    emits a built-in code gets that code's remedy from the remedy planner.

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
    :return: The classified VASP run report; its ``process.argv`` is the wrapper command line when dispatching.
    :raises ValueError: If a command, or a configured ``vasp.ncl_command``/``vasp.gamma_command``,
        starts with a launcher while a launch prefix applies, if *policy* is invalid, or if a checker
        follows another file inside the run directory, or one there that is a symlink or not a regular file.
    """

    # Parsed first: an invalid policy must not touch the workdir.
    diagnostic_policy = parse_diagnostic_policy(policy, checks=checks)
    root = Path(directory).resolve()
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
    # A stale source would replay the previous run into the checks; one outside the run directory is not ours.
    for source in follow:
        if source.path.is_relative_to(root):
            source.path.unlink(missing_ok=True)
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
    supervisor = ProcessSupervisor(
        monitors=(_VaspMonitor(checks, diagnostic_policy), *monitors), checkers=checkers, follow=follow
    )
    process = supervisor.run(
        command,
        timeout=timeout,
        cwd=root,
        termination_grace=termination_grace,
        stdout_path=root / "vasp.out",
        stderr_path=root / "vasp.err",
    )
    diagnostics = _deduplicate((*process.diagnostics, *diagnose_vasp_files(root)))
    if process.timed_out:
        classification = "timeout"
    elif any(item.stop for item in diagnostics):
        classification = "diagnosed_stop"
    elif process.returncode:
        classification = "process_failure"
    elif any(
        item.code in {"electronic_nonconvergence", "ionic_nonconvergence", "positive_final_energy"}
        for item in diagnostics
    ):
        classification = "nonconverged"
    elif any(item.code == "incomplete_outcar" for item in diagnostics):
        classification = "process_failure"
    else:
        classification = "completed"
    report = VaspRunReport(process, classification, diagnostics, diagnostic_policy)
    report.write(root / report_path)
    return report


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
