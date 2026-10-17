"""Choose and exec the VASP binary variant a directory's inputs need.

A launcher such as custodian runs one configured VASP command, but whether a
calculation is non-collinear or Gamma-only is only known once its ``INCAR`` and
``KPOINTS`` exist. Pointing that command at
``python -m httk.codes.vasp.launch --standard CMD [--ncl CMD] [--gamma CMD]``
behind the launch prefix (``srun … python -m httk.codes.vasp.launch …``) makes
every rank read the inputs in its working directory and exec the right binary.
The commands travel in argv, not the environment, because a confined launch
drops ``HTTK_``-prefixed variables from rank environments. Importing this module
imports the package :mod:`httk.codes.vasp` and through it *httk-core* and numpy,
about 0.3 s per rank; pymatgen is never imported.
"""

import argparse
import os
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from .inputs import read_incar

__all__ = ["main", "select_vasp_command"]


def _true(value: str | None) -> bool:
    """Return whether an INCAR value is a true Fortran logical (``.TRUE.``, ``T``, ...)."""

    return value is not None and value.strip().lstrip(".")[:1].upper() == "T"


def _gamma_only_mesh(path: Path) -> bool:
    """Return whether ``path`` is an automatic Gamma/Monkhorst-Pack ``1 1 1`` mesh without shift."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if lines[2].strip()[:1].lower() not in {"g", "m"} or [int(item) for item in lines[3].split()] != [1, 1, 1]:
            return False
        shift = [float(item) for item in lines[4].split()[:3]] if len(lines) > 4 else []
    except (OSError, IndexError, ValueError):
        # No KPOINTS (KSPACING), or a fully automatic, explicit, line-mode or malformed one, is not a 1 1 1 mesh;
        # a non-numeric fifth line counts as an absent shift, as in pymatgen.
        return False
    return all(abs(value) < 1.0e-6 for value in shift)


def select_vasp_command(
    directory: str | os.PathLike[str] = ".",
    *,
    standard: Sequence[str],
    ncl: Sequence[str] | None = None,
    gamma: Sequence[str] | None = None,
) -> list[str]:
    """Select the VASP command variant for the inputs in ``directory``.

    ``LNONCOLLINEAR`` or ``LSORBIT`` true in ``INCAR`` selects ``ncl``.
    Otherwise ``gamma`` is selected, when configured, for a Gamma-only run as
    custodian's ``_gamma_point_only_check`` judges it: neither ``LEPSILON`` nor
    ``LOPTICS`` true, and ``KPOINTS`` an automatic Gamma or Monkhorst-Pack mesh
    of ``1 1 1`` whose shift is absent or zero. Unlike custodian, a missing
    ``KPOINTS`` (a ``KSPACING`` run) selects ``standard``. Every other case
    selects ``standard``.

    :param directory: Read ``INCAR`` and ``KPOINTS`` from this directory.
    :param standard: The standard VASP command (``vasp_std``).
    :param ncl: The non-collinear VASP command (``vasp_ncl``), if configured.
    :param gamma: The Gamma-only VASP command (``vasp_gam``), if configured.
    :return: A copy of the selected command.
    :raises ValueError: If the run is non-collinear and ``ncl`` is not configured.
    :raises OSError: If ``INCAR`` cannot be read.
    """

    return _select(directory, standard, ncl, gamma)[1]


def _select(
    directory: str | os.PathLike[str], standard: Sequence[str], ncl: Sequence[str] | None, gamma: Sequence[str] | None
) -> tuple[str, list[str]]:
    """Return the selected variant name with a copy of its command, as :func:`select_vasp_command` selects."""

    root = Path(directory)
    incar = read_incar(root / "INCAR")
    if _true(incar.get("LNONCOLLINEAR")) or _true(incar.get("LSORBIT")):
        if ncl is None:
            raise ValueError(
                "INCAR sets LNONCOLLINEAR or LSORBIT but no non-collinear VASP command is configured "
                "(set vasp.ncl_command, or pass --ncl)"
            )
        return "ncl", list(ncl)
    if (
        gamma is not None
        and not any(_true(incar.get(tag)) for tag in ("LEPSILON", "LOPTICS"))
        and _gamma_only_mesh(root / "KPOINTS")
    ):
        return "gamma", list(gamma)
    return "standard", list(standard)


def _command(value: str) -> list[str]:
    """Split one shlex-quoted command argument, refusing an empty one."""

    command = shlex.split(value)
    if not command:
        raise argparse.ArgumentTypeError("a command cannot be empty")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Select the VASP command for a directory and exec it.

    :param argv: Parse these arguments instead of ``sys.argv[1:]``.
    :return: ``2`` when no command can be selected or executed; on success the
        process is replaced and this never returns.
    """

    parser = argparse.ArgumentParser(
        prog="python -m httk.codes.vasp.launch",
        description="Exec the standard, non-collinear or Gamma-only VASP command the INCAR and KPOINTS need.",
    )
    parser.add_argument("--standard", required=True, type=_command, help="standard VASP command (shlex string)")
    parser.add_argument("--ncl", type=_command, help="non-collinear VASP command (shlex string)")
    parser.add_argument("--gamma", type=_command, help="Gamma-only VASP command (shlex string)")
    parser.add_argument(
        "--directory",
        default=".",
        help="read INCAR and KPOINTS from this directory; the command still runs in the current one (default: .)",
    )
    arguments = parser.parse_args(argv)
    try:
        variant, command = _select(arguments.directory, arguments.standard, arguments.ncl, arguments.gamma)
    except (OSError, ValueError) as error:
        print(f"httk.codes.vasp.launch: {error}", file=sys.stderr)
        return 2
    try:
        os.execvp(command[0], command)
    except OSError as error:
        print(
            f"httk.codes.vasp.launch: cannot execute {command[0]!r} (the --{variant} command): {error.strerror or error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
