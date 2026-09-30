"""Collect hook for a finished VASP relaxation found in a directory.

It reads POSCAR, CONTCAR and OUTCAR from the directory, each possibly compressed.
"""

from httk.workflow.collecting import JobRecord

from httk.codes.vasp.collect import read_structure, read_total_energy, structure_precision


def collect(record: JobRecord) -> dict[str, object]:
    """Return the input structure, the relaxed structure, and the final energy.

    :param record: The collected job record.
    :return: Extracted roles for the relaxation.
    """
    precision = structure_precision(record.result_file("OUTCAR").parent)
    return {
        "initial_structure": read_structure(record.result_file("POSCAR"), precision=precision),
        "relaxed_structure": read_structure(record.result_file("CONTCAR"), precision=precision),
        "total_energy": read_total_energy(record.result_file("OUTCAR")),
    }
