"""The default VASP launch wrapper: choose the binary, plan parallelization, exec.

A launcher such as custodian runs one configured VASP command, but whether a
calculation is non-collinear or Gamma-only, and how its ranks are best split
over k-points and bands, is only known once its ``INCAR``, ``POSCAR``,
``POTCAR`` and ``KPOINTS`` exist. Pointing that command at
``python -m httk.codes.vasp.launch --standard CMD [--ncl CMD] [--gamma CMD]``
runs this module once per launch, in front of the attempt's launch prefix: it
selects the binary, writes the ``KPAR``/``NCORE`` that
:func:`httk.codes.vasp.parallel.plan_parallelization` plans into the INCAR
unless the INCAR already sets them, records the decision on stderr and in
``httk-vasp-launch.log``, and execs the launch prefix (``HTTK_WORKFLOW_LAUNCH``)
followed by the selected binary. It never runs per rank, so the INCAR is
rewritten by exactly one process before any rank reads it; pymatgen is never
imported. :func:`httk.codes.vasp.run_vasp` routes through it by default.
"""

import argparse
import json
import math
import os
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from httk.workflow.codes import launch_command, utc_now

from .inputs import _true, read_incar, update_incar
from .parallel import plan_parallelization

__all__ = ["main", "select_vasp_command"]


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


def _positive(value: str) -> int:
    """Parse a positive integer option."""

    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1:
        raise argparse.ArgumentTypeError(f"not a positive integer: {value!r}")
    return number


def _context() -> dict[str, object]:
    """Return the attempt context from ``HTTK_WORKFLOW_CONTEXT``, or an empty mapping outside an attempt."""

    try:
        value = json.loads(os.environ.get("HTTK_WORKFLOW_CONTEXT") or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _setting(name: str) -> object:
    """Resolve one setting as ``Attempt.setting`` does: job parameter, ``HTTK_`` variable, workspace setting."""

    job = os.environ.get("HTTK_WORKFLOW_JOB_DIR")
    if job:
        try:
            parameters = json.loads((Path(job) / "job.json").read_text(encoding="utf-8")).get("parameters")
        except (OSError, ValueError, AttributeError):
            parameters = None
        if isinstance(parameters, dict) and name in parameters:
            return parameters[name]
    variable = "HTTK_" + name.upper().replace(".", "_")
    if variable in os.environ:
        return os.environ[variable]
    settings = _context().get("settings")
    return settings.get(name) if isinstance(settings, dict) else None


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


#: Launcher flags that decide the rank layout: total ranks, ranks per node, OpenMP threads per rank.
_RANK_FLAGS = {
    "-n": "total",
    "-np": "total",
    "--ntasks": "total",
    "--ntasks-per-node": "per_node",
    "--cpus-per-task": "threads",
    "OMP_NUM_THREADS": "omp",
}


def _launcher_flags(argv: Sequence[str]) -> dict[str, int]:
    """Return the first ``total``, ``per_node``, ``threads`` and ``omp`` values a launcher's flags give."""

    names = dict(_RANK_FLAGS)
    # -c is srun's --cpus-per-task, but Open MPI's mpirun reads it as -np.
    names["-c"] = "threads" if any(os.path.basename(word) == "srun" for word in argv) else "total"
    found: dict[str, int] = {}
    for index, word in enumerate(argv):
        name, equals, text = word.partition("=")
        if not equals:
            name, text = word, argv[index + 1] if index + 1 < len(argv) else ""
        if name in names and text.removeprefix("-").isdigit():
            found.setdefault(names[name], int(text))
    return found


def _attempt_ranks(
    prefix: Sequence[str], command: Sequence[str], context: Mapping[str, object]
) -> tuple[int | None, int | None, str]:
    """Return the launch's rank count, its ranks per node when known, and where they came from.

    The launcher's own flags (the prefix's, or without a prefix the command's)
    win: ``-n``/``-np``/``--ntasks``, else ``--ntasks-per-node`` times the
    binding's node count. With a prefix and no such flag, the attempt context
    says what the prefix starts: the binding's per-node ``procs``, else the
    resource requirement ``procs``. More than one OpenMP thread per rank (an
    explicit ``OMP_NUM_THREADS``, in the prefix or the environment, else
    ``--cpus-per-task``/srun's ``-c``), a thread flag without a derivable total,
    a count below 1, and a binding with GPUs leave the count unknown, so nothing
    is planned.
    """

    flags = _launcher_flags(prefix or command)
    if any(value < 1 for value in flags.values()):
        return None, None, "a launcher count below 1"
    environment_threads = os.environ.get("OMP_NUM_THREADS", "")
    if "omp" not in flags and environment_threads.isdigit():
        flags["omp"] = int(environment_threads)
    # An explicit OMP_NUM_THREADS is what VASP runs with; cpus per task only implies threads without it.
    threads = flags.get("omp", flags.get("threads", 1))
    if threads > 1:
        return None, None, f"{threads} OpenMP threads per rank"
    procs: list[int] = []
    if prefix:
        binding = context.get("binding")
        nodes = binding.get("nodes") if isinstance(binding, dict) else None
        if isinstance(nodes, list) and nodes:
            if any(isinstance(node, dict) and _count(node.get("gpus")) for node in nodes):
                return None, None, "the attempt binding has GPUs"
            counts = [_count(node.get("procs")) if isinstance(node, dict) else None for node in nodes]
            procs = [item for item in counts if item is not None]
            if len(procs) != len(counts):
                procs = []
    if "total" in flags:
        per_node = flags.get("per_node") or (math.gcd(*procs) if procs and sum(procs) == flags["total"] else None)
        return flags["total"], per_node, "the launcher's rank flag"
    if "per_node" in flags:
        if procs:
            return flags["per_node"] * len(procs), flags["per_node"], "--ntasks-per-node times the binding's nodes"
        return None, None, "--ntasks-per-node without a node count"
    if "threads" in flags:
        return None, None, "a thread flag without a rank count"
    if procs:
        return sum(procs), math.gcd(*procs), "the attempt binding"
    resources = context.get("resources") if prefix else None
    procs_value = _count(resources.get("procs")) if isinstance(resources, dict) else None
    if procs_value is not None:
        return procs_value, None, "the attempt resources"
    return None, None, "no binding, resources or launcher rank flag"


def _parallel_lines(
    root: Path, mode: str, ranks_option: int | None, prefix: Sequence[str], command: Sequence[str]
) -> list[str]:
    """Plan and apply the parallelization tags, returning the decision and warning lines."""

    if mode == "off":
        return ["parallel: off (vasp.parallelization); INCAR unchanged"]
    ranks, per_node, source = (
        (ranks_option, None, "--ranks") if ranks_option is not None else _attempt_ranks(prefix, command, _context())
    )
    if ranks is None:
        return [f"parallel: rank count unknown ({source}); INCAR unchanged"]
    try:
        plan = plan_parallelization(root, ranks=ranks, ranks_per_node=per_node)
        tags = {name: value for name, value in (("KPAR", plan.kpar), ("NCORE", plan.ncore)) if value is not None}
        if tags:
            update_incar(tags, root / "INCAR")
    except (OSError, ValueError) as error:
        return [f"parallel: not planned ({error}); INCAR unchanged"]
    written = " ".join(f"{name}={value}" for name, value in tags.items())
    return [
        f"parallel: {plan.reason}, ranks from {source}; " + (f"wrote {written}" if tags else "INCAR unchanged"),
        *(f"warning: {item}" for item in plan.warnings),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Select the VASP command for a directory, plan its parallelization, and exec it behind the launch prefix.

    :param argv: Parse these arguments instead of ``sys.argv[1:]``.
    :return: ``2`` when no command can be selected or executed; on success the
        process is replaced and this never returns.
    """

    parser = argparse.ArgumentParser(
        prog="python -m httk.codes.vasp.launch",
        description="Exec the standard, non-collinear or Gamma-only VASP command the inputs need, behind the "
        "attempt's launch prefix, after writing planned KPAR/NCORE into the INCAR.",
    )
    parser.add_argument("--standard", required=True, type=_command, help="standard VASP command (shlex string)")
    parser.add_argument("--ncl", type=_command, help="non-collinear VASP command (shlex string)")
    parser.add_argument("--gamma", type=_command, help="Gamma-only VASP command (shlex string)")
    parser.add_argument(
        "--directory",
        default=".",
        help="read and update the inputs in this directory; the command still runs in the current one (default: .)",
    )
    parser.add_argument(
        "--parallel",
        choices=("auto", "off"),
        help="plan KPAR/NCORE (auto) or leave the INCAR alone (off); default: the vasp.parallelization setting, "
        "else auto",
    )
    parser.add_argument("--ranks", type=_positive, help="plan for this many MPI ranks instead of the attempt's")
    parser.add_argument("--no-launch", action="store_true", help="exec the command without the launch prefix")
    arguments = parser.parse_args(argv)
    try:
        variant, command = _select(arguments.directory, arguments.standard, arguments.ncl, arguments.gamma)
        launched = launch_command(command, launch=not arguments.no_launch)
        value = arguments.parallel or _setting("vasp.parallelization")
        mode = "auto" if value is None else value
        if mode not in ("auto", "off"):
            raise ValueError(f"vasp.parallelization must be 'auto' or 'off', not {mode!r}")
    except (OSError, ValueError) as error:
        print(f"httk.codes.vasp.launch: {error}", file=sys.stderr)
        return 2
    root = Path(arguments.directory)
    lines = _parallel_lines(root, str(mode), arguments.ranks, launched[: len(launched) - len(command)], command)
    try:
        with (root / "httk-vasp-launch.log").open("a", encoding="utf-8") as log:
            log.writelines(f"{utc_now()} {line}\n" for line in lines)
    except OSError as error:
        lines.append(f"warning: cannot append to httk-vasp-launch.log: {error.strerror or error}")
    for line in lines:
        print(f"httk.codes.vasp.launch: {line}", file=sys.stderr, flush=True)
    try:
        os.execvp(launched[0], launched)
    except OSError as error:
        what = f"the --{variant} command" if len(launched) == len(command) else "the launch prefix"
        print(
            f"httk.codes.vasp.launch: cannot execute {launched[0]!r} ({what}): {error.strerror or error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
