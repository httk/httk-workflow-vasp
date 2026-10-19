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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from httk.workflow.codes import (
    Diagnostic,
    FollowSource,
    ProcessReport,
    ProcessSupervisor,
    SourceEvent,
    launch_command,
    write_json_atomic,
)

from .diagnostics import _VASP_PATTERNS, diagnose_vasp_files


class _VaspMonitor:
    def __call__(self, event: SourceEvent) -> Sequence[Diagnostic]:
        if event.event != "line" or event.line is None:
            return ()
        result: list[Diagnostic] = []
        allocation = re.search(r"total allocation\s*:\s*([0-9]+)\s*KBytes", event.line, re.IGNORECASE)
        if allocation is not None and int(allocation.group(1)) > 500_000:
            result.append(
                Diagnostic(
                    "realspace_allocation_too_large",
                    "fatal",
                    "VASP requested more than 500000 KBytes for real-space projection",
                    event.source,
                    event.line,
                    True,
                )
            )
        for pattern, code, severity, stop in _VASP_PATTERNS:
            if pattern.search(event.line):
                result.append(
                    Diagnostic(
                        code,
                        severity,  # type: ignore[arg-type]
                        event.line.strip(),
                        event.source,
                        event.line,
                        stop,
                    )
                )
        return result


@dataclass(frozen=True)
class VaspRunReport:
    """Classified result of one supervised VASP execution.

    :param process: Preserve the supervised process result.
    :param classification: Record the final VASP run classification.
    :param diagnostics: Preserve live and file-level diagnostics.
    """

    process: ProcessReport
    classification: str
    diagnostics: tuple[Diagnostic, ...]

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
    :return: The classified VASP run report; its ``process.argv`` is the wrapper command line when dispatching.
    :raises ValueError: If a command, or a configured ``vasp.ncl_command``/``vasp.gamma_command``,
        starts with a launcher while a launch prefix applies.
    """

    root = Path(directory).resolve()
    follow = (
        FollowSource(root / "OSZICAR", "OSZICAR"),
        FollowSource(root / "OUTCAR", "OUTCAR"),
    )
    for source in follow:
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
    supervisor = ProcessSupervisor(monitors=(_VaspMonitor(),), follow=follow)
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
    report = VaspRunReport(process, classification, diagnostics)
    report.write(root / report_path)
    return report


def _deduplicate(values: Sequence[Diagnostic]) -> tuple[Diagnostic, ...]:
    result: list[Diagnostic] = []
    seen: set[tuple[str, str, str | None]] = set()
    for item in values:
        key = item.code, item.source, item.evidence
        if key not in seen:
            seen.add(key)
            result.append(item)
    return tuple(result)
