"""End-of-job finalization of VASP outputs: OUTCAR cleaning, POTCAR summaries, compression.

A succeeded VASP job is sealed with its outputs compressed, its OUTCAR stripped
of the per-k-point eigenvalue and k-point listing blocks, and every POTCAR (a
copyrighted file that must not be redistributed) replaced by a header summary
plus the SHA-256 of the original. This ports the end-of-job archival of *httk*
v1 (``VASP_CLEAN_OUTCAR``, ``VASP_POTCAR_SUMMARY``, ``HT_TASK_COMPRESS``).
Every step is atomic and idempotent, so rerunning :func:`finalize_outputs`
after a crash at any point completes the job. The ``find_output`` family reads
a file whether or not it has been compressed since.
"""

import hashlib
import logging
import os
import platform
import re
import shutil
import socket
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import IO, TYPE_CHECKING, cast

from httk.core.datastream.compression import codec_for_name, open_compressed, split_compression_suffix
from httk.workflow.runtime_utils import compress_files

from .inputs import _TEMPORARY_SUFFIX, _potcar_summary_text, _umask, _write_bytes_atomic, _write_text_atomic

if TYPE_CHECKING:
    from httk.workflow import Attempt

logger = logging.getLogger(__name__)

#: The compression methods finalization accepts; ``"none"`` (or ``None``) skips compression.
COMPRESSION_METHODS: tuple[str, ...] = ("zst", "bz2", "gz", "xz", "none")

# The v1 VASP_CLEAN_OUTCAR awk program (vasptools.sh), rule by rule. Mode 1 (the
# per-k-point occupation blocks) ends at a dashed line, mode 2 (k-point listings)
# at a blank line; the ending line is kept.
_OCCUPATION = re.compile(rb"^ *k-point *[0-9*]* *: +[0-9.-]* +[0-9.-]* +[0-9.-]* *$")
_LISTINGS = (
    (re.compile(rb"^ *k-points in units of 2pi/SCALE and weight:"), b"VASP_CLEAN_OUTCAR: Removed k-point listning.\n"),
    (re.compile(rb"^ *Following reciprocal coordinates:"), b"VASP_CLEAN_OUTCAR: Removed k-point specification\n"),
    (re.compile(rb"^ *Following cartesian coordinates:"), b"VASP_CLEAN_OUTCAR: Removed k-point specification\n"),
    (
        re.compile(rb"^ *k-points in reciprocal lattice and weights"),
        b"VASP_CLEAN_OUTCAR: Removed k-point specification\n",
    ),
    (re.compile(rb"^ *k-point +[0-9*]* .* plane waves"), b"VASP_CLEAN_OUTCAR: Removed k-point plane wave data\n"),
)
_DASHES = re.compile(rb"^ *-----------------------------------")
# The temporaries a crash can leave below a root: compress_files' ``.<NAME>.<ext>.<pid>.tmp``
# and the atomic writers' ``.<NAME>.<8 tempfile characters>.httk-tmp``.
_STALE_TEMPORARY = re.compile(r"\..+\.(?:bz2|gz|xz|zst)\.[0-9]+\.tmp|\..+\.[a-z0-9_]{8}" + re.escape(_TEMPORARY_SUFFIX))
_BLANK = re.compile(rb"^ *$")


def _replace_atomic(path: Path, write: Callable[[IO[bytes]], object]) -> None:
    """Atomically replace ``path`` with what ``write`` streams, keeping an existing file's mode."""

    try:
        mode = path.stat().st_mode & 0o7777
    except FileNotFoundError:
        mode = 0o666 & ~_umask()
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=_TEMPORARY_SUFFIX, dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            write(stream)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def clean_outcar(path: str | os.PathLike[str] = "OUTCAR") -> Path:
    """Remove the per-k-point eigenvalue/occupation blocks and k-point listings from OUTCAR in place.

    The output is byte-for-byte what *httk* v1's ``VASP_CLEAN_OUTCAR`` awk
    program writes, including its marker lines (and their historic spelling), so
    v2 archives match v1 ones. The file is streamed and replaced atomically;
    cleaning a cleaned OUTCAR changes nothing. Energies, structures, forces,
    stresses, magnetization, parameters and the completion footer are kept.

    :param path: Clean the OUTCAR file at this path.
    :return: The cleaned OUTCAR path.
    :raises FileNotFoundError: If the file does not exist.
    """

    source = Path(path)

    def write(output: IO[bytes]) -> None:
        mode = 0
        with source.open("rb") as lines:
            for raw in lines:
                line = raw[:-1] if raw.endswith(b"\n") else raw
                if mode == 0:
                    if _OCCUPATION.match(line):
                        output.write(b"VASP_CLEAN_OUTCAR: Removed k-point occupation data\n")
                        mode = 1
                    else:
                        for pattern, marker in _LISTINGS:
                            if pattern.match(line):
                                output.write(marker)
                                mode = 2
                                break
                if (mode == 1 and _DASHES.match(line)) or (mode == 2 and _BLANK.match(line)):
                    mode = 0
                if mode == 0:
                    output.write(line + b"\n")

    if not source.is_file():
        raise FileNotFoundError(source)
    _replace_atomic(source, write)
    return source


def summarize_potcar(path: str | os.PathLike[str] = "POTCAR") -> Path:
    """Replace a POTCAR by its header summary and SHA-256.

    Writes ``POTCAR.summary`` (the :func:`~httk.codes.vasp.potcar_summary`
    format) and ``POTCAR.sha256`` (``sha256sum`` format, so ``sha256sum -c
    POTCAR.sha256`` verifies a reassembled POTCAR) next to it, then removes the
    POTCAR. Each write is atomic. A compressed ``POTCAR.<suffix>`` is summarized
    and checksummed by its decompressed bytes. A symlink is summarized by what
    it points to, and only the link is removed; a dangling one is removed with a
    warning. A missing POTCAR beside an existing summary means the work is
    already done.

    :param path: Summarize and remove the POTCAR (or compressed POTCAR, or symlink to one) at this path.
    :return: The summary path.
    :raises FileNotFoundError: If neither the POTCAR nor its summary exists.
    """

    source = Path(path)
    name = split_compression_suffix(source.name)[0]
    summary = source.with_name(name + ".summary")
    if not source.is_file():
        if source.is_symlink():
            logger.warning("removing dangling POTCAR symlink %s", source)
            source.unlink()
            return summary
        if summary.is_file():
            return summary
        raise FileNotFoundError(source)
    with source.open("rb") as raw, open_compressed(raw, name=source.resolve().name) as stream:
        data = cast(bytes, stream.read())
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    _write_text_atomic(summary, _potcar_summary_text(text))
    _write_bytes_atomic(source.with_name(name + ".sha256"), f"{hashlib.sha256(data).hexdigest()}  {name}\n".encode())
    source.unlink()
    return summary


def default_compress_workers(context: Mapping[str, object] | None = None) -> int:
    """Return how many compression workers the attempt's allocation affords on this host.

    The binding node whose ``host`` is this host (``socket.gethostname()``,
    ``platform.node()`` or ``SLURMD_NODENAME``, compared by their first DNS
    label) gives its ``procs``. Without
    binding nodes, the resource requirement's ``procs`` is used, else every CPU
    this process may run on. A binding that excludes this host gives 1. The
    result is capped at the CPU-affinity count and is at least 1.

    :param context: The attempt context; ``None`` reads ``HTTK_WORKFLOW_CONTEXT``.
    :return: The worker count.
    """

    # Deferred: launch runs as ``python -m httk.codes.vasp.launch``, so the package must not import it.
    from .launch import _context, _count

    context = _context() if context is None else context
    available = len(os.sched_getaffinity(0))
    binding = context.get("binding")
    nodes = binding.get("nodes") if isinstance(binding, dict) else None
    nodes = [node for node in nodes if isinstance(node, dict)] if isinstance(nodes, list) else []

    def label(host: object) -> str:
        return str(host).split(".")[0].lower()  # so a short name matches its FQDN

    names = (socket.gethostname(), platform.node(), os.environ.get("SLURMD_NODENAME"))
    hosts = {label(host) for host in names if host}
    if nodes:
        mine = [node for node in nodes if node.get("host") and label(node.get("host")) in hosts]
        workers = (_count(mine[0].get("procs")) or available) if mine else 1
    else:
        resources = context.get("resources")
        workers = (_count(resources.get("procs")) if isinstance(resources, dict) else None) or available
    return max(1, min(workers, available))


def _entries(roots: Iterable[str | os.PathLike[str]]) -> list[Path]:
    """Return every non-directory entry below ``roots``, symlinks included, outside dot directories."""

    found: list[Path] = []
    for root in roots:
        for directory, subdirectories, names in os.walk(root):
            subdirectories[:] = [name for name in subdirectories if not name.startswith(".")]
            found += [Path(directory, name) for name in names]
    return found


def _walk(roots: Iterable[str | os.PathLike[str]]) -> list[Path]:
    """Return the regular, non-dot files below ``roots``, without following symlinks."""

    return [path for path in _entries(roots) if not path.name.startswith(".") and _regular(path)]


def _regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _method(compress: str | None) -> str | None:
    if compress is None or compress == "none":
        return None
    if compress not in COMPRESSION_METHODS:
        raise ValueError(f"compression must be one of {', '.join(COMPRESSION_METHODS)}, got {compress!r}")
    return compress


_clean_outcar = clean_outcar
_summarize_potcar = summarize_potcar


def finalize_outputs(
    *roots: str | os.PathLike[str],
    compress: str | None = "zst",
    level: int | None = None,
    workers: int | None = None,
    clean_outcar: bool = True,
    summarize_potcar: bool = True,
    compress_roots: Sequence[str | os.PathLike[str]] | None = None,
) -> None:
    """Finalize a job's outputs in place: clean OUTCARs, summarize POTCARs, then compress.

    Each root is walked recursively without following symlinks, skipping dot
    files and dot directories. Every regular file named ``OUTCAR`` below the
    output roots ``compress_roots`` is cleaned (:func:`clean_outcar`); every
    ``POTCAR`` below any root, compressed or a symlink, is summarized
    (:func:`summarize_potcar`); and then every remaining regular file below the
    output roots whose name has no registered compression suffix is compressed,
    replacing the original and any older compressed copy of it. So a root that
    is not an output root, such as a job's staged ``files/``, only loses its
    POTCARs. Rerunning after a crash at any point completes the work: a file
    left beside its compressed copy is compressed again, and the partial
    temporaries the crash left are removed first (only names of exactly the
    form these writers use; other dot files are kept).

    :param \\*roots: Finalize the files below these directories.
    :param compress: Compress with ``zst``, ``bz2``, ``gz`` or ``xz``; ``None`` or ``"none"`` skips compression.
    :param level: Select the compression level; ``None`` uses the codec default.
    :param workers: Bound the compression parallelism; ``None`` uses :func:`default_compress_workers`.
    :param clean_outcar: Clean every OUTCAR below the output roots.
    :param summarize_potcar: Replace every POTCAR by its summary and SHA-256.
    :param compress_roots: The output roots, cleaned and compressed; ``None`` means all ``roots``.
    :raises ValueError: If a compression option is invalid; nothing is changed then.
    """

    method = _method(compress)
    if method is not None:
        workers = default_compress_workers() if workers is None else workers
        compress_files((), method=method, level=level, workers=workers)  # validates before anything changes
    outputs = roots if compress_roots is None else compress_roots
    for path in _entries(roots):  # partial temporaries a crashed finalization left
        if _STALE_TEMPORARY.fullmatch(path.name) and _regular(path):
            path.unlink()
    if clean_outcar:
        for path in _walk(outputs):
            if path.name == "OUTCAR":
                _clean_outcar(path)
    if summarize_potcar:
        for path in _entries(roots):
            if split_compression_suffix(path.name)[0] == "POTCAR" and (path.is_symlink() or path.is_file()):
                _summarize_potcar(path)
    if method is None:
        return
    files = _walk(outputs)
    targets = [path for path in files if codec_for_name(path.name) is None]
    # A compressed sibling of a file about to be compressed is stale and would shadow the new copy in
    # find_output (which prefers the registry's earlier codecs); the plain file still holds the data.
    fresh = set(targets)
    for path in files:
        inner, codec = split_compression_suffix(path.name)
        if codec is not None and path.with_name(inner) in fresh and path.name != f"{inner}.{method}":
            path.unlink()
    compress_files(targets, method=method, level=level, workers=cast(int, workers), remove_source=True)


def _bool_setting(name: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).lower() if isinstance(value, str | int) else None
    if token in ("true", "yes", "1"):
        return True
    if token in ("false", "no", "0"):
        return False
    raise ValueError(f"setting {name} must be true or false (or yes/no/1/0), got {value!r}")


def _int_setting(name: str, value: object, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or (minimum is not None and value < minimum):
        floor = "" if minimum is None else f" of at least {minimum}"
        raise ValueError(f"setting {name} must be an integer{floor}, got {value!r}")
    return value


def _setting(a: "Attempt", name: str, default: object) -> object:
    """Return a setting, with JSON ``null`` or the string ``"null"`` meaning unset."""

    value = a.setting(name, default)
    return default if value is None or value == "null" else value


def finalize_job(a: "Attempt") -> None:
    """Finalize a succeeded VASP job's outputs before it publishes and seals them.

    Cleans OUTCAR, summarizes every POTCAR and compresses everything in the
    workdir, as :func:`finalize_outputs`; the POTCAR summary also applies to the
    payload's ``files/``, where nothing else changes. The options are read
    through :meth:`~httk.workflow.Attempt.setting`:

    ==========================  ==========================================  ==============
    Setting                     Values                                      Default
    ==========================  ==========================================  ==============
    ``vasp.compress``           ``zst``, ``bz2``, ``gz``, ``xz``, ``none``  ``zst``
    ``vasp.compress_level``     integer                                     codec default
    ``vasp.compress_workers``   positive integer                            automatic
    ``vasp.clean_outcar``       boolean                                     true
    ``vasp.summarize_potcar``   boolean                                     true
    ==========================  ==========================================  ==============

    Booleans also accept the strings true/false/yes/no/1/0. A JSON ``null`` or
    the string ``"null"`` means unset, so the default applies.

    :param a: The running attempt.
    :raises ValueError: If a setting value is invalid, naming the setting; nothing is changed then.
    """

    compress = _setting(a, "vasp.compress", "zst")
    if not isinstance(compress, str) or compress not in COMPRESSION_METHODS:
        raise ValueError(f"setting vasp.compress must be one of {', '.join(COMPRESSION_METHODS)}, got {compress!r}")
    level = _int_setting("vasp.compress_level", _setting(a, "vasp.compress_level", None))
    workers = _int_setting("vasp.compress_workers", _setting(a, "vasp.compress_workers", None), 1)
    if compress != "none":
        try:
            compress_files((), method=compress, level=level)
        except ValueError as exc:
            raise ValueError(f"setting vasp.compress_level: {exc}") from None
    finalize_outputs(
        a.workdir,
        a.payload / "files",
        compress=compress,
        level=level,
        workers=workers,
        clean_outcar=_bool_setting("vasp.clean_outcar", _setting(a, "vasp.clean_outcar", True)),
        summarize_potcar=_bool_setting("vasp.summarize_potcar", _setting(a, "vasp.summarize_potcar", True)),
        compress_roots=(a.workdir,),
    )


def find_output(directory: str | os.PathLike[str], name: str) -> Path | None:
    """Find an output file or its compressed copy (``OUTCAR`` or ``OUTCAR.zst``, ...).

    :param directory: Look in this directory.
    :param name: The uncompressed file name.
    :return: The existing file, or ``None`` when neither it nor a compressed copy exists.
    """

    # Deferred: httk.workflow.collecting is heavy, and every Bash bridge call imports this package.
    from httk.workflow.collecting import existing_file

    return existing_file(Path(directory) / name)


def _read(path: Path) -> bytes:
    """Return the decompressed bytes of ``path``, or of its compressed copy when it is missing."""

    found = find_output(path.parent, path.name) or path
    with found.open("rb") as raw, open_compressed(raw, compression="extension", name=found.name) as stream:
        return cast(bytes, stream.read())


def _read_text(path: Path, errors: str = "strict") -> str:
    """Return :func:`_read` decoded as UTF-8 with universal newlines, as :meth:`pathlib.Path.read_text` reads."""

    return _read(path).decode("utf-8", errors).replace("\r\n", "\n").replace("\r", "\n")


def read_output_bytes(directory: str | os.PathLike[str], name: str) -> bytes:
    """Return an output file's decompressed bytes, whether or not it was compressed.

    :param directory: Look in this directory.
    :param name: The uncompressed file name.
    :return: The file's contents.
    :raises FileNotFoundError: If neither the file nor a compressed copy exists.
    """

    return _read(Path(directory) / name)


def read_output_text(directory: str | os.PathLike[str], name: str, encoding: str = "utf-8") -> str:
    """Return an output file's decompressed text, whether or not it was compressed.

    :param directory: Look in this directory.
    :param name: The uncompressed file name.
    :param encoding: Decode with this encoding.
    :return: The file's text.
    :raises FileNotFoundError: If neither the file nor a compressed copy exists.
    """

    return read_output_bytes(directory, name).decode(encoding)


def copy_output(directory: str | os.PathLike[str], name: str, destination: str | os.PathLike[str]) -> Path:
    """Copy an output file, decompressed, to a plain ``destination``, atomically.

    :param directory: Look in this directory.
    :param name: The uncompressed file name.
    :param destination: Write the decompressed copy to this path.
    :return: The destination path.
    :raises FileNotFoundError: If neither the file nor a compressed copy exists.
    """

    source = Path(directory) / name
    found = find_output(source.parent, source.name)
    if found is None:
        raise FileNotFoundError(source)
    target = Path(destination)

    def write(output: IO[bytes]) -> None:
        with found.open("rb") as raw, open_compressed(raw, compression="extension", name=found.name) as stream:
            shutil.copyfileobj(cast(IO[bytes], stream), output)

    _replace_atomic(target, write)
    return target
