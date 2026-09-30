"""Recognize a finished VASP static run in a directory."""

from pathlib import Path

from httk.workflow.calculations import content_digest
from httk.workflow.hookapi import Claim, Unclaimed

from httk.codes.vasp.collect import classify_vasp_calculation


def recognize(directory: Path) -> Claim | Unclaimed | None:
    """Claim the directory when its OUTCAR echoes a static run.

    :param directory: The directory holding OUTCAR and POSCAR.
    :return: A claim identified by the input files, an ``Unclaimed``, or ``None``.
    """
    kind = classify_vasp_calculation(directory)
    if isinstance(kind, Unclaimed):
        return kind
    if kind != "static":
        return None
    return Claim(content_digest(directory, ["INCAR", "POSCAR", "KPOINTS", "POTCAR"]))
