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
from pathlib import Path
from typing import Any

from httk.core import DataRecord

__all__ = ["read_structure", "read_total_energy"]

_TOTAL_ENERGY_DEFINITION = "https://schemas.httk.org/defs/v0.1/properties/core/total_energy"
_TOTAL_ENERGY_NAME = "_httk_total_energy"


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
