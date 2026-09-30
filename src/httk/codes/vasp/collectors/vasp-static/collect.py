"""Collect hook for a finished VASP static run found in a directory.

It reads POSCAR and OUTCAR from the directory, each possibly compressed.
"""

from httk.workflow.collecting import JobRecord

from httk.codes.vasp.collect import read_structure, read_total_energy


def collect(record: JobRecord) -> dict[str, object]:
    """Return the input structure and the final energy.

    :param record: The collected job record.
    :return: Extracted roles for the static run.
    """
    return {
        "initial_structure": read_structure(record.result_file("POSCAR")),
        "total_energy": read_total_energy(record.result_file("OUTCAR")),
    }
