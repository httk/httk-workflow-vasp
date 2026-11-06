"""VASP output diagnosis, report reading, and workdir cleanup.

This is the read-side of the dependency-free VASP interface: extracting single
values from OSZICAR/vasprun/OUTCAR, the diagnostic pattern table that classifies
VASP-5/6 failure lines, the file-level completion diagnosis, and the workdir
validation and cleanup helpers. Historical authorship is documented in
``v1_runtime/NOTICE``.
"""

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from httk.workflow.codes import Diagnostic

from .inputs import _write_text_atomic

#: Outputs :func:`clean_vasp_outputs` keeps unless they are named explicitly:
#: the remedy machinery and restart promotion read exactly these two files.
VASP_RESTART_ARTIFACTS: tuple[str, ...] = ("CONTCAR", "vasp-run-report.json")


def last_oszicar_energy(path: str | os.PathLike[str] = "OSZICAR") -> float | None:
    """Return the final ``E0`` value, or ``None`` when it is absent.

    :param path: Read the OSZICAR file at this path.
    :return: The final parsed ``E0`` value, or ``None`` when no value is found.
    """

    found: float | None = None
    pattern = re.compile(r"\bE0=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)")
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match is not None:
            found = float(match.group(1))
    return found


def last_vasprun_volume(path: str | os.PathLike[str] = "vasprun.xml") -> float | None:
    """Return the final volume reported by ``vasprun.xml``.

    :param path: Read the vasprun XML file at this path.
    :return: The final parsed volume, or ``None`` when no volume is found.
    """

    found: float | None = None
    pattern = re.compile(r'<i\s+name=["\']volume["\']>\s*([-+0-9.Ee]+)\s*</i>')
    for match in pattern.finditer(Path(path).read_text(encoding="utf-8", errors="replace")):
        found = float(match.group(1))
    return found


def outcar_potim(path: str | os.PathLike[str] = "OUTCAR") -> float | None:
    """Return the last optimizer step size from OUTCAR.

    :param path: Read the OUTCAR file at this path.
    :return: The last parsed optimizer step size, or ``None`` when absent.
    """

    found: float | None = None
    pattern = re.compile(r"^\s*opt step\s*=\s*([-+0-9.Ee]+)", re.MULTILINE)
    for match in pattern.finditer(Path(path).read_text(encoding="utf-8", errors="replace")):
        found = float(match.group(1))
    return found


def outcar_plane_wave_count(path: str | os.PathLike[str] = "OUTCAR") -> int | None:
    """Return OUTCAR's maximum plane-wave count.

    :param path: Read the OUTCAR file at this path.
    :return: The parsed maximum plane-wave count, or ``None`` when absent.
    """

    match = re.search(
        r"^\s*maximum number of plane-waves\s*:\s*([0-9]+)",
        Path(path).read_text(encoding="utf-8", errors="replace"),
        re.MULTILINE,
    )
    return None if match is None else int(match.group(1))


def clean_outcar(
    path: str | os.PathLike[str] = "OUTCAR",
    output: str | os.PathLike[str] = "OUTCAR.cleaned",
) -> Path:
    """Remove the largest reproducible k-point detail blocks from OUTCAR.

    :param path: Read the OUTCAR file at this path.
    :param output: Write the cleaned OUTCAR to this path.
    :return: The cleaned output path.
    """

    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    result: list[str] = []
    skipping = False
    starts = (
        "k-points in units of 2pi/SCALE and weight:",
        "Following reciprocal coordinates:",
        "Following cartesian coordinates:",
        "k-points in reciprocal lattice and weights",
    )
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(item) for item in starts) or re.match(r"^k-point\s+\d+\s+.*plane waves", stripped):
            result.append("VASP_CLEAN_OUTCAR: removed k-point detail block")
            skipping = True
            continue
        if skipping:
            if not stripped:
                skipping = False
            continue
        result.append(line)
    destination = Path(output)
    _write_text_atomic(destination, "\n".join(result) + "\n")
    return destination


def validate_vasp_workdir(
    path: str | os.PathLike[str] = ".",
    *,
    maximum_length: int = 240,
) -> Path:
    """Validate VASP's conservative absolute-path length constraint.

    :param path: Resolve and validate this workdir path.
    :param maximum_length: Reject resolved paths longer than this byte length.
    :return: The resolved workdir path.
    :raises ValueError: If the limit is invalid or the resolved path is too long.
    """

    resolved = Path(path).resolve()
    if maximum_length < 1:
        raise ValueError("maximum path length must be positive")
    if len(os.fspath(resolved).encode()) > maximum_length:
        raise ValueError(f"VASP workdir path exceeds {maximum_length} bytes: {resolved}")
    return resolved


def clean_vasp_outputs(
    directory: str | os.PathLike[str] = ".",
    *,
    keep: Iterable[str] = (),
    also_remove: Iterable[str] = (),
) -> tuple[Path, ...]:
    """Remove standard rerun outputs while preserving declared names.

    The files in :data:`VASP_RESTART_ARTIFACTS` — ``CONTCAR`` and
    ``vasp-run-report.json`` — are kept even without *keep*, because they are what
    the remedy machinery and restart promotion read: a pre-run cleanup that
    deletes them destroys the evidence of the run it is cleaning up after. Name
    them in *also_remove* to delete them anyway.

    :param directory: Remove outputs from this directory.
    :param keep: Preserve these output names.
    :param also_remove: Allow restart artifacts to be removed by naming them here.
    :return: The paths removed from the directory.
    :raises ValueError: If an output to remove is not a regular file.
    """

    root = Path(directory)
    retained = frozenset(keep) | (frozenset(VASP_RESTART_ARTIFACTS) - frozenset(also_remove))
    names = (
        "CHG",
        "CHGCAR",
        "CONTCAR",
        "DOSCAR",
        "EIGENVAL",
        "IBZKPT",
        "OSZICAR",
        "OUTCAR",
        "PCDAT",
        "PROCAR",
        "REPORT",
        "vasprun.xml",
        "WAVECAR",
        "XDATCAR",
        "vasp.out",
        "vasp.err",
        "vasp-run-report.json",
    )
    removed: list[Path] = []
    for name in names:
        path = root / name
        if name in retained or not path.exists():
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"refusing to clean non-regular VASP output: {path}")
        path.unlink()
        removed.append(path)
    return tuple(removed)


@dataclass(frozen=True)
class VaspCheck:
    """One live check :func:`~httk.codes.vasp.run_vasp` applies to VASP output lines.

    :param code: Name the diagnostic the check emits; the remedy planner selects a remedy by this code.
    :param pattern: Match an OUTCAR, OSZICAR, stdout or stderr line with this expression.
    :param severity: Classify the emitted diagnostic.
    :param stop: Terminate the run when the check matches.
    """

    code: str
    pattern: re.Pattern[str]
    severity: Literal["info", "warning", "error", "fatal"]
    stop: bool


#: The live checks :func:`~httk.codes.vasp.run_vasp` applies to every OUTCAR,
#: OSZICAR, stdout and stderr line. Besides them it applies the numeric
#: ``realspace_allocation_too_large`` check (a real-space projection allocation
#: above 500000 KBytes, a stopping ``fatal`` diagnostic).
VASP_CHECKS: tuple[VaspCheck, ...] = (
    VaspCheck("chgcar_incomplete", re.compile(r"chargedensity file is incomplete", re.IGNORECASE), "error", True),
    VaspCheck("zpotrf", re.compile(r"ZPOTRF failed", re.IGNORECASE), "fatal", True),
    VaspCheck("fexcf", re.compile(r"FEXCF: supplied exchange-correlation table", re.IGNORECASE), "error", True),
    VaspCheck(
        "kpoints_class",
        re.compile(r"Reciprocal lattice and k-lattice belong to different class", re.IGNORECASE),
        "error",
        True,
    ),
    VaspCheck(
        "tetrahedron_kpoints",
        re.compile(r"Tetrahedron method fails|Fatal error.*k-mesh|unable to match k-point|TETIRR needs", re.IGNORECASE),
        "error",
        True,
    ),
    VaspCheck(
        "inverse_rotation", re.compile(r"inverse of rotation matrix was not found", re.IGNORECASE), "error", True
    ),
    VaspCheck(
        "rotation_matrix",
        re.compile(r"Found some non-integer element in rotation matrix", re.IGNORECASE),
        "error",
        True,
    ),
    VaspCheck("brmix", re.compile(r"BRMIX: very serious problems", re.IGNORECASE), "warning", False),
    VaspCheck("kpoint_shifts", re.compile(r"Could not get correct shifts", re.IGNORECASE), "warning", False),
    VaspCheck("real_optlay", re.compile(r"REAL_OPTLAY: internal error", re.IGNORECASE), "error", True),
    VaspCheck("rspher", re.compile(r"(?:internal ERROR RSPHER|RSPHER: internal ERROR)", re.IGNORECASE), "fatal", True),
    VaspCheck("dentet", re.compile(r"DENTET: can't reach specified precision", re.IGNORECASE), "warning", False),
    VaspCheck("too_few_bands", re.compile(r"TOO FEW BANDS", re.IGNORECASE), "error", False),
    VaspCheck("triple_product", re.compile(r"triple product of the basis vectors", re.IGNORECASE), "fatal", True),
    VaspCheck("brions", re.compile(r"BRIONS problems: POTIM should be increased", re.IGNORECASE), "warning", False),
    VaspCheck("aliasing", re.compile(r"small aliasing.*errors", re.IGNORECASE), "warning", False),
    VaspCheck("ions_too_close", re.compile(r"distance between some ions is very small", re.IGNORECASE), "fatal", True),
    VaspCheck("lreal_false", re.compile(r"set LREAL=.FALSE", re.IGNORECASE), "error", True),
    VaspCheck(
        "pricell", re.compile(r"number of cells and number of vectors did not agree", re.IGNORECASE), "error", True
    ),
    VaspCheck("radint", re.compile(r"internal error in RAD_INT", re.IGNORECASE), "fatal", True),
    VaspCheck("nonlr_alloc", re.compile(r"internal ERROR in NONLR_ALLOC", re.IGNORECASE), "fatal", True),
    # VASP 5/6 use EDDDAV or EDDAV, with or without "Error", also inside a box.
    # Keep the historical diagnostic code so existing consumers/history work.
    VaspCheck(
        "edddav_zhegv",
        re.compile(r"\b(?:Error\s+)?EDD?DAV\s*:\s*Call\s+to\s+ZHEGV\s+failed\b", re.IGNORECASE),
        "error",
        False,
    ),
    VaspCheck("cnormn", re.compile(r"CNORMN: search vector ill defined", re.IGNORECASE), "warning", False),
    VaspCheck("zbrent_bracketing", re.compile(r"ZBRENT: fatal error in bracketing", re.IGNORECASE), "error", False),
    VaspCheck(
        "lattice_vector_too_long", re.compile(r"One of the lattice vectors is very long", re.IGNORECASE), "fatal", True
    ),
)


#: The live check :mod:`~httk.codes.vasp.reports` applies besides :data:`VASP_CHECKS`.
_ALLOCATION_CODE = "realspace_allocation_too_large"
_POLICY_ACTIONS = frozenset({"stop", "continue", "ignore"})
#: Codes only :func:`diagnose_vasp_files` produces, after the run.
_POST_RUN_CODES = frozenset(
    {"incomplete_outcar", "electronic_nonconvergence", "ionic_nonconvergence", "positive_final_energy"}
)


def parse_diagnostic_policy(value: object, *, checks: Sequence[VaspCheck] = VASP_CHECKS) -> dict[str, str]:
    """Parse a per-code policy for the live VASP checks.

    The policy maps a live diagnostic code to ``stop`` (end the run when it
    matches), ``continue`` (report it but let the run go on) or ``ignore`` (drop
    it entirely, so the remedy planner never sees it either): ``{"brmix": "stop"}``,
    or the space-separated string ``"brmix=stop zpotrf=continue"`` that Bash
    runners and workspace settings carry.

    :param value: Parse this mapping, ``CODE=ACTION`` string, or ``None``/``""`` for no policy.
    :param checks: Accept the codes of these checks, besides ``realspace_allocation_too_large``.
    :return: The validated code to action mapping.
    :raises ValueError: If the value or one of its entries is malformed, a code
        repeats or is unknown, a code is one the post-run file diagnosis produces,
        or an action is not ``stop``, ``continue`` or ``ignore``.
    """

    if value is None or value == "":
        return {}
    pairs: list[tuple[object, object]] = []
    if isinstance(value, str):
        for token in value.split():
            name, separator, choice = token.partition("=")
            if not separator:
                raise ValueError(f"diagnostic_policy entry {token!r} is not of the form CODE=ACTION")
            pairs.append((name, choice))
    elif isinstance(value, Mapping):
        pairs.extend(value.items())
    else:
        raise ValueError(
            f"diagnostic_policy must be a mapping or a 'CODE=ACTION ...' string, not {type(value).__name__}"
        )
    known = {check.code for check in checks} | {_ALLOCATION_CODE}
    policy: dict[str, str] = {}
    for code, action in pairs:
        if not isinstance(code, str) or not isinstance(action, str) or not code or not action:
            raise ValueError(f"diagnostic_policy entry {code!r}={action!r} needs a nonempty code and action")
        if code in policy:
            raise ValueError(f"diagnostic_policy names code {code!r} more than once")
        if code not in known:
            if code in _POST_RUN_CODES:
                # Accepting a nonconverged or incomplete run as completed is deliberately not offered.
                raise ValueError(
                    f"diagnostic_policy entry {code!r} is a post-run file diagnosis; the policy governs live checks only"
                )
            raise ValueError(f"diagnostic_policy names unknown code {code!r}; known codes: {', '.join(sorted(known))}")
        if action not in _POLICY_ACTIONS:
            raise ValueError(f"diagnostic_policy entry {code}={action} is not one of stop, continue, ignore")
        policy[code] = action
    return policy


def diagnose_vasp_files(directory: str | os.PathLike[str] = ".") -> tuple[Diagnostic, ...]:
    """Diagnose completion and final convergence from VASP output files.

    :param directory: Read VASP output files from this directory.
    :return: Diagnostics inferred from the available output files.
    """

    root = Path(directory)
    diagnostics: list[Diagnostic] = []
    outcar = root / "OUTCAR"
    oszicar = root / "OSZICAR"
    outcar_text = outcar.read_text(encoding="utf-8", errors="replace") if outcar.is_file() else ""
    if (
        re.search(
            r"General timing and accounting information(?:s)? for this job:",
            outcar_text,
            re.IGNORECASE,
        )
        is None
    ):
        diagnostics.append(Diagnostic("incomplete_outcar", "error", "OUTCAR has no normal completion footer", "OUTCAR"))
    nelm_match = re.search(r"^\s*NELM\s*=\s*([0-9]+)", outcar_text, re.MULTILINE)
    nsw_match = re.search(r"^\s*NSW\s*=\s*([0-9]+)", outcar_text, re.MULTILINE)
    nelm = None if nelm_match is None else int(nelm_match.group(1))
    nsw = None if nsw_match is None else int(nsw_match.group(1))
    if oszicar.is_file():
        electronic_step = 0
        ionic_steps = 0
        last_energy: float | None = None
        for line in oszicar.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"^[A-Za-z]+:\s+([0-9]+)\s+", line)
            if match:
                electronic_step = int(match.group(1))
            ionic = re.match(r"^\s*[0-9]+\s+F=\s*([-+0-9.Ee]+)", line)
            if ionic:
                ionic_steps += 1
                last_energy = float(ionic.group(1))
        if nelm is not None and electronic_step >= nelm:
            diagnostics.append(
                Diagnostic("electronic_nonconvergence", "error", "final electronic step reached NELM", "OSZICAR")
            )
        if nsw is not None and nsw > 1 and ionic_steps >= nsw:
            diagnostics.append(Diagnostic("ionic_nonconvergence", "error", "ionic steps reached NSW", "OSZICAR"))
        if last_energy is not None and last_energy > 0:
            diagnostics.append(Diagnostic("positive_final_energy", "error", "final free energy is positive", "OSZICAR"))
    return tuple(diagnostics)
