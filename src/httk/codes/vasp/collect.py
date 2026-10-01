"""Building blocks for the collect hooks of VASP workflows.

A workflow's ``collect.py`` decides which files of a finished job hold its
outputs; these helpers read one result out of such a file. Files are located
with ``record.result_file`` of ``httk.workflow.collecting.JobRecord``::

    prefix = record.parameter("data_prefix", "vasp") or ""
    return {
        "relaxed_structure": read_structure(record.result_file("CONTCAR", data_prefix=prefix)),
        "total_energy": read_total_energy(record.result_file("OUTCAR", data_prefix=prefix)),
    }
"""

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from httk.core import DataRecord
from httk.workflow.hookapi import Unclaimed

__all__ = ["classify_vasp_calculation", "read_structure", "read_total_energy", "structure_precision"]

_TOTAL_ENERGY_DEFINITION = "https://schemas.httk.org/defs/v0.1/properties/core/total_energy"
_TOTAL_ENERGY_NAME = "_httk_total_energy"
_DEFAULT_STRUCTURE_PRECISION = 0.001  # VASP's default EDIFFG magnitude, in Å


def _load(path: Path, *, raw: bool = False, precision: float | None = None) -> Any:
    try:
        importlib.import_module("httk.atomistic")
        core = importlib.import_module("httk.core")
        if not core.has_reader_for(path.name):
            raise ImportError(f"no reader is registered for {path.name}")
        return core.load(str(path), raw=raw, **({} if precision is None else {"precision": precision}))
    except ImportError as exc:
        raise ValueError(
            f"VASP collecting requires the file readers and structure adapters provided by httk-atomistic: {exc}"
        ) from exc


def read_structure(path: Path, *, precision: float | None = None) -> object:
    """Read a structure from a VASP structure file such as a CONTCAR.

    :param path: The file, whose name selects the reader.
    :param precision: The coordinate precision in Å, passed to the reader; when ``None``
        the reader infers it from the digits written and warns.
    :return: A ``httk.atomistic.UnitcellStructureView`` of the structure.
    :raises ValueError: If the file cannot be read as a structure.
    """

    try:
        loaded = _load(path, precision=precision)
        atomistic = importlib.import_module("httk.atomistic")
        return atomistic.UnitcellStructureView(loaded)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot construct a structure from {path}: {exc}") from exc


def read_total_energy(path: Path) -> DataRecord:
    """Read the final total energy, extrapolated to zero smearing, from an OUTCAR.

    :param path: The OUTCAR.
    :return: The energy, in eV, as a ``total_energy`` property record.
    :raises ValueError: If the OUTCAR is incomplete or has no final energy.
    """

    lexeme: Any = None
    try:
        outcar = _load(path, raw=True)["outcar"]
        if not outcar.completed:
            raise ValueError(f"{path} is incomplete: VASP completion footer is missing")
        final_energies = getattr(outcar, "final_energies", None)
        lexeme = None if final_energies is None else getattr(final_energies, "energy_sigma0", None)
        return DataRecord.from_value(_TOTAL_ENERGY_DEFINITION, _TOTAL_ENERGY_NAME, float(lexeme))
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot construct a total energy from {path}; energy_sigma0={lexeme!r}: {exc}") from exc


def _echoed_parameters(directory: Path) -> Mapping[str, str] | Unclaimed:
    """Return the parameters echoed by the OUTCAR of a directory, or why there is no OUTCAR."""

    from httk.atomistic.integrations.vasp.io import OutcarFile
    from httk.workflow.collecting import existing_file

    path = existing_file(Path(directory) / "OUTCAR")
    if path is None:
        return Unclaimed("no OUTCAR")
    with OutcarFile(path) as outcar:
        return outcar.parameters


def _nsw_ibrion(parameters: Mapping[str, str]) -> tuple[int, int] | None:
    try:
        return int(parameters["NSW"]), int(parameters["IBRION"])
    except (KeyError, ValueError):
        return None


def classify_vasp_calculation(directory: Path) -> str | Unclaimed:
    """Classify a VASP calculation directory from the parameters its OUTCAR echoes.

    :param directory: The directory holding the OUTCAR, possibly compressed.
    :return: ``"relax"`` (``NSW > 0`` with ``IBRION`` 1, 2 or 3), ``"static"``
        (``NSW == 0`` or ``IBRION == -1``), or an ``Unclaimed`` saying why the
        directory cannot be collected yet (molecular dynamics, another IBRION,
        or an OUTCAR without a parameter echo).
    """

    parameters = _echoed_parameters(directory)
    if isinstance(parameters, Unclaimed):
        return parameters
    steps = _nsw_ibrion(parameters)
    if steps is None:
        return Unclaimed("OUTCAR has no parameter echo")
    nsw, ibrion = steps
    if ibrion == 0:
        return Unclaimed("molecular dynamics (IBRION = 0) is not collected yet")
    if nsw == 0 or ibrion == -1:
        return "static"
    if ibrion in {1, 2, 3}:
        return "relax"
    return Unclaimed(f"IBRION = {ibrion} is not collected yet")


def structure_precision(directory: Path) -> float:
    """Return the precision, in Å, for reading the POSCAR and CONTCAR of a calculation.

    The precision is taken from the calculation's own convergence settings, as
    echoed by its OUTCAR: the magnitude of ``EDIFFG`` for a relaxation (``NSW > 0``
    and ``IBRION != -1``), otherwise the magnitude of ``EDIFF``. A zero or
    unparseable value counts as absent (``EDIFFG = 0`` means "use ``EDIFF``"). With
    no OUTCAR or no usable parameters, it is 0.001, VASP's default ``EDIFFG`` magnitude.

    :param directory: The directory holding the OUTCAR, possibly compressed.
    :return: The precision in Å, always greater than zero.
    """

    parameters = _echoed_parameters(directory)
    if isinstance(parameters, Unclaimed):
        return _DEFAULT_STRUCTURE_PRECISION
    steps = _nsw_ibrion(parameters)
    relaxing = steps is not None and steps[0] > 0 and steps[1] != -1
    for key in ("EDIFFG", "EDIFF") if relaxing else ("EDIFF",):
        try:
            value = abs(float(parameters[key].replace("D", "E").replace("d", "e")))
        except (KeyError, ValueError):
            continue
        if value > 0:
            return value
    return _DEFAULT_STRUCTURE_PRECISION
