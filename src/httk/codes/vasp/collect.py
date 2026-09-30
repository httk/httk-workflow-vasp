"""Building blocks for the collect hooks of VASP workflows.

A workflow's ``collect.py`` decides which files of a finished job hold its
outputs; these helpers locate one such file and read one result out of it::

    prefix = job_parameter(record, "data_prefix", "vasp")
    return {
        "relaxed_structure": read_structure(result_file(record, "CONTCAR", data_prefix=prefix)),
        "total_energy": read_total_energy(result_file(record, "OUTCAR", data_prefix=prefix)),
    }
"""

import importlib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from httk.core import DataRecord
from httk.workflow.collecting import JobRecord

__all__ = ["job_parameter", "read_structure", "read_total_energy", "result_file"]

_TOTAL_ENERGY_DEFINITION = "https://schemas.httk.org/defs/v0.1/properties/core/total_energy"
_TOTAL_ENERGY_NAME = "_httk_total_energy"


def _identity(record: JobRecord) -> str:
    return f"{record.workspace_id}:{record.job_id}"


def job_parameter(record: JobRecord, name: str, default: str) -> str:
    """Return one string parameter of the job, or a default.

    :param record: The collected job record.
    :param name: The parameter name.
    :param default: The value when the job does not set the parameter to a string.
    :return: The parameter value.
    """

    parameters = record.job.get("parameters")
    value = parameters.get(name) if isinstance(parameters, Mapping) else None
    return value if isinstance(value, str) else default


def result_file(record: JobRecord, name: str, *, data_prefix: str = "", published: str | None = None) -> Path:
    """Locate one result file of a finished job.

    A job with transactional data is read from its committed data, at
    ``published`` (default ``name``) below ``data_prefix``; any other job is
    read from its persistent workdir, at ``name``. A transactional job without
    committed data fails rather than falling back to unpublished workdir files.

    :param record: The collected job record.
    :param name: The file's path relative to the workdir.
    :param data_prefix: The directory below the job's data the runner published under.
    :param published: The file's path below ``data_prefix``, when the runner
        published it under another name than it has in the workdir.
    :return: The absolute path of the existing file.
    :raises ValueError: If the job has no such file.
    """

    if record.data is not None:
        relative = str(PurePosixPath(data_prefix, published or name))
        if record.data_generation is None:
            raise ValueError(
                f"{_identity(record)}: expected published data file {relative!r}, but the job has no published data"
            )
        path = record.data / relative
        if not path.is_file():
            raise ValueError(f"{_identity(record)}: expected published data file {path}")
        return path
    if record.workdir is None:
        raise ValueError(f"{_identity(record)}: expected workdir file {name!r}, but the job has no workdir")
    path = record.workdir / name
    if not path.is_file():
        raise ValueError(f"{_identity(record)}: expected workdir file {path}")
    return path


def _load(path: Path, *, raw: bool = False) -> Any:
    try:
        importlib.import_module("httk.atomistic")
        core = importlib.import_module("httk.core")
        if not core.has_reader_for(path.name):
            raise ImportError(f"no reader is registered for {path.name}")
        return core.load(str(path), raw=raw)
    except ImportError as exc:
        raise ValueError(
            f"VASP collecting requires the file readers and structure adapters provided by httk-atomistic: {exc}"
        ) from exc


def read_structure(path: Path) -> object:
    """Read a structure from a VASP structure file such as a CONTCAR.

    :param path: The file, whose name selects the reader.
    :return: A ``httk.atomistic.UnitcellStructureView`` of the structure.
    :raises ValueError: If the file cannot be read as a structure.
    """

    try:
        loaded = _load(path)
        atomistic = importlib.import_module("httk.atomistic")
        return atomistic.UnitcellStructureView(loaded)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot construct a structure from {path}: {exc}") from exc


def read_total_energy(path: Path) -> DataRecord:
    """Read the final total energy, extrapolated to zero smearing, from an OUTCAR.

    :param path: The OUTCAR.
    :return: The energy, in eV, as a ``total_energy`` property record.
    :raises ValueError: If the OUTCAR has no final energy.
    """

    lexeme: Any = None
    try:
        final_energies = getattr(_load(path, raw=True)["outcar"], "final_energies", None)
        lexeme = None if final_energies is None else getattr(final_energies, "energy_sigma0", None)
        return DataRecord.from_value(_TOTAL_ENERGY_DEFINITION, _TOTAL_ENERGY_NAME, float(lexeme))
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot construct a total energy from {path}; energy_sigma0={lexeme!r}: {exc}") from exc
