"""Plan VASP's ``KPAR``, ``NCORE`` and ``NBANDS`` from a calculation's inputs and rank count.

VASP splits its ranks into ``KPAR`` k-point groups and each group into
``NPAR = ranks / KPAR / NCORE`` band groups, and rounds ``NBANDS`` up to a
multiple of ``NPAR``. Left at VASP's defaults (``KPAR = 1``, ``NCORE = 1``), a
small cell on many ranks gets one band group per rank and a band count far above
what it asked for. For an ``NBANDS`` the INCAR sets, VASP reports the change and
custodian refuses the run as unrecoverable once the rounded value exceeds twice
``NELECT``; VASP's own default count is rounded silently, which only wastes
time. :func:`plan_parallelization` chooses the two
tags just before VASP starts, when every input and the rank count are known; the
launch wrapper :mod:`httk.codes.vasp.launch` writes them into the INCAR and then,
unless ``vasp.nbands_rounding`` is ``off``, rounds ``NBANDS`` itself: an explicit
value up to a multiple of ``NPAR``, an absent one to VASP's default made explicit
and divisible, in both cases only while the result stays at most ``2 NELECT``. A
value it writes carries the comment ``# httk: rounded from BASE for NPAR N``
(``BASE`` is ``default M`` for VASP's default), so a later launch on other ranks
rounds from ``BASE`` again instead of ratcheting up; an edited value loses that
status and counts as explicit.
"""

import itertools
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .inputs import (
    PoscarHeader,
    _expanded_sum,
    _reciprocal_lengths,
    _true,
    _updated_incar,
    _valence_electrons,
    read_incar,
    read_poscar_header,
)

__all__ = ["ParallelPlan", "plan_parallelization"]

#: The NCORE VASP's documentation recommends on most machines; candidates closest to it win.
_PREFERRED_NCORE = 8
#: VASP's KSPACING when neither a KPOINTS file nor the tag is present.
_DEFAULT_KSPACING = 0.5
#: An NBANDS line the launch wrapper wrote: value, marker comment, ``default ``, base, NPAR.
_MARKER = re.compile(
    r"^[ \t]*NBANDS[ \t]*=[ \t]*(\d+)[ \t]*(#[ \t]*httk: rounded from (default )?(\d+) for NPAR ([1-9]\d*))[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
#: A marker comment on a line of its own.
_BARE_MARKER = re.compile(r"^[ \t]*#[ \t]*httk: rounded from.*(?:\n|\Z)", re.MULTILINE)


@dataclass(frozen=True)
class ParallelPlan:
    """The parallelization tags planned for one VASP launch.

    :param ranks: The rank count the plan is for.
    :param kpar: The ``KPAR`` to write, or ``None`` to leave the INCAR's (VASP's default ``1``).
    :param ncore: The ``NCORE`` to write, or ``None`` to leave the INCAR's (VASP's default ``1``).
    :param reason: One human-readable line explaining the decision.
    :param warnings: Problems the plan could not avoid, one line each.
    """

    ranks: int
    kpar: int | None
    ncore: int | None
    reason: str
    warnings: tuple[str, ...] = ()


def _divisors(value: int) -> list[int]:
    return [item for item in range(1, value + 1) if value % item == 0]


def _rounded(nbands: int, npar: int) -> int:
    """Return ``nbands`` rounded up to a multiple of ``npar``, as VASP rounds it."""

    return -(-nbands // npar) * npar


def _nelect(root: Path, incar: Mapping[str, str], header: PoscarHeader) -> float:
    if "NELECT" in incar:
        return float(incar["NELECT"])
    return _valence_electrons(header, root / "POTCAR")


def _determinant(matrix: Sequence[Sequence[int]]) -> int:
    return sum(
        matrix[0][k]
        * (matrix[1][(k + 1) % 3] * matrix[2][(k + 2) % 3] - matrix[1][(k + 2) % 3] * matrix[2][(k + 1) % 3])
        for k in range(3)
    )


def _lattice_point_group(header: PoscarHeader) -> list[tuple[tuple[int, ...], ...]]:
    """Return the lattice's point group as integer matrices ``N`` with ``N G N^T = G`` for the POSCAR metric ``G``.

    As spglib does, the basis is first Delaunay (Selling) reduced and the three
    shortest independent vectors of the superbase and its pair sums taken; the
    search over entries 0, ±1 then runs in that basis, and the matrices are
    mapped back to the POSCAR basis. The tolerance is relative to the largest
    metric entry, so it is loose for a strongly anisotropic cell; that only
    admits extra operations, which lowers the orbit count, so it stays a lower bound.
    """

    metric = [[sum(a * b for a, b in zip(u, v, strict=True)) for v in header.lattice] for u in header.lattice]
    scale = max(abs(item) for row in metric for item in row)

    def product(r: Sequence[int], s: Sequence[int], metric: list[list[float]] = metric) -> float:
        return sum(r[i] * metric[i][j] * s[j] for i in range(3) for j in range(3))

    # Selling reduction of the superbase b1..b4 (b4 = -(b1 + b2 + b3)), as integer combinations of the POSCAR rows.
    base = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]]
    pairs = list(itertools.combinations(range(4), 2))
    # Capped for a (near-)singular lattice: a partly reduced basis still gives a subgroup, so a lower bound.
    for _ in range(1000):
        pair = next(((i, j) for i, j in pairs if product(base[i], base[j]) > 1e-8 * scale), None)
        if pair is None:
            break
        first = base[pair[0]]
        for k in set(range(4)) - set(pair):
            base[k] = [a + b for a, b in zip(base[k], first, strict=True)]
        base[pair[0]] = [-a for a in first]
    vectors = base + [[a + b for a, b in zip(base[i], base[j], strict=True)] for i, j in ((0, 1), (0, 2), (1, 2))]
    vectors.sort(key=lambda vector: product(vector, vector))
    transform = next(rows for rows in itertools.combinations(vectors, 3) if abs(_determinant(rows)) == 1)
    reduced = [[product(u, v) for v in transform] for u in transform]
    tolerance = 1e-5 * scale

    rows = list(itertools.product((-1, 0, 1), repeat=3))
    candidates = [[r for r in rows if abs(product(r, r, reduced) - reduced[i][i]) <= tolerance] for i in range(3)]
    found = [
        op
        for op in itertools.product(*candidates)
        if all(
            abs(product(op[i], op[j], reduced) - reduced[i][j]) <= tolerance for i in range(3) for j in range(i + 1, 3)
        )
    ]
    # N = T^-1 N_red T, with T^-1 the integer adjugate times det T = ±1.
    sign = _determinant(transform)
    inverse = [
        [
            sign
            * (
                transform[(j + 1) % 3][(i + 1) % 3] * transform[(j + 2) % 3][(i + 2) % 3]
                - transform[(j + 1) % 3][(i + 2) % 3] * transform[(j + 2) % 3][(i + 1) % 3]
            )
            for j in range(3)
        ]
        for i in range(3)
    ]

    def times(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3)) for i in range(3))

    return [times(times(inverse, op), transform) for op in found]


def _mesh_bound(header: PoscarHeader, mesh: tuple[int, ...], gamma: bool, limit: int | None) -> int:
    """Count the lattice point group's orbits that meet the mesh, stopping once ``limit`` is reached.

    The crystal's point group is a subgroup of its lattice's, and VASP's
    irreducible points each lie in one such orbit, so the count bounds VASP's
    from below whether VASP drops operations that do not map the whole mesh onto
    itself or identifies points one by one. Points are integer triples over the
    common denominator ``2 prod(mesh)``; a k-point ``κ`` maps to ``N κ``.
    """

    if len(mesh) != 3 or min(mesh) < 1:
        raise ValueError(f"invalid k-point mesh {mesh}")
    denominator = 2 * math.prod(mesh)
    steps = [denominator // n for n in mesh]
    # Monkhorst-Pack points are (2m - n + 1) / 2n; the step is even, so the shift is exact.
    origin = [0 if gamma else (1 - n) * step // 2 for n, step in zip(mesh, steps, strict=True)]
    ops = _lattice_point_group(header)
    seen: set[tuple[int, ...]] = set()
    count = 0
    for indices in itertools.product(*map(range, mesh)):
        point = tuple((o + k * s) % denominator for o, k, s in zip(origin, indices, steps, strict=True))
        if point in seen:
            continue
        count += 1
        if count == limit:
            break
        # Images outside the mesh are marked too; they are never scanned.
        seen.update(tuple(sum(a * b for a, b in zip(row, point)) % denominator for row in op) for op in ops)
    return count


def _kpoint_bound(
    root: Path, incar: Mapping[str, str], header: PoscarHeader, *, limit: int | None = None
) -> tuple[int, str]:
    """Return a lower bound on the irreducible k-point count, counted up to ``limit``, and how it was derived."""

    path = root / "KPOINTS"
    if not path.exists():
        spacing = float(incar.get("KSPACING", _DEFAULT_KSPACING))
        mesh = tuple(max(1, math.ceil(length * 2 * math.pi / spacing)) for length in _reciprocal_lengths(header))
        gamma = _true(incar.get("KGAMMA", ".TRUE."))
        return _mesh_bound(header, mesh, gamma, limit), f"KSPACING={spacing:g} mesh {'x'.join(map(str, mesh))}"
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        count = int(lines[1].split()[0])
        style = lines[2].strip()[:1].lower()
        if count == 0 and style in {"g", "m"}:
            mesh = tuple(int(item) for item in lines[3].split()[:3])
            if len(lines) > 4 and any(float(item) for item in lines[4].split()[:3]):
                # ponytail: a shifted mesh falls back to its size over the largest Laue-group order (48);
                # count shifted orbits if such meshes need a tighter KPAR.
                return max(1, math.ceil(math.prod(mesh) / 48)), f"mesh {'x'.join(map(str, mesh))} shifted"
            return _mesh_bound(header, mesh, style == "g", limit), f"mesh {'x'.join(map(str, mesh))}"
        if count == 0 and style == "a":
            # A fully automatic file: VASP's N_i = int(max(1, R_k |b_i| + 0.5)).
            length = float(lines[3].split()[0])
            mesh = tuple(max(1, int(length * item + 0.5)) for item in _reciprocal_lengths(header))
            source = f"automatic length {length:g} mesh {'x'.join(map(str, mesh))}"
            return _mesh_bound(header, mesh, True, limit), source
        if count > 0 and style == "l":
            segments = sum(1 for line in lines[4:] if re.split(r"[!#]", line, maxsplit=1)[0].split()) // 2
            return max(1, count * segments), f"line mode, {segments} segments"
        if count > 0:
            return count, "explicit list"
    except (IndexError, ValueError):
        pass
    return 1, "unrecognized KPOINTS"


def _default_nbands(nelect: float, nions: int, incar: Mapping[str, str]) -> int:
    """Return VASP's default NBANDS before rounding to NPAR.

    This is VASP's ``main.F`` formula, which differs from the one on the VASP
    wiki, as pymatgen's ``VaspInputSet.estimate_nbands`` documents:
    ``max(floor((NELECT + 2) / 2) + max(floor(NIONS / 2), 3), floor(0.6 NELECT))``,
    plus ``floor((sum(MAGMOM) + 1) / 2)`` for ``ISPIN = 2`` (``MAGMOM`` defaults
    to 1 per ion), doubled for a non-collinear run.
    """

    nbands = max(math.floor((nelect + 2) / 2) + max(nions // 2, 3), math.floor(0.6 * nelect))
    if int(float(incar.get("ISPIN", "1"))) == 2:
        nbands += math.floor((_expanded_sum(incar.get("MAGMOM", f"{nions}*1")) + 1) / 2)
    if _true(incar.get("LNONCOLLINEAR")) or _true(incar.get("LSORBIT")):
        nbands *= 2
    return nbands


def _marker(text: str, incar: Mapping[str, str]) -> re.Match[str] | None:
    """Return the launch wrapper's NBANDS marker when the INCAR still holds the value it wrote, else ``None``."""

    found = list(_MARKER.finditer(text))
    if not found or "NBANDS" not in incar:
        return None
    match = found[-1]
    value = int(match[1])
    if value != int(float(incar["NBANDS"])) or value != _rounded(int(match[4]), int(match[5])):
        return None
    return match


def _requested_nbands(text: str, incar: Mapping[str, str]) -> int | None:
    """Return the NBANDS the INCAR asks for: a value the wrapper rounded counts as its base, ``None`` as VASP's default."""

    match = _marker(text, incar)
    if match is not None:
        return None if match[3] else int(match[4])
    return int(float(incar["NBANDS"])) if "NBANDS" in incar else None


def _band_groups(incar: Mapping[str, str], ranks: int) -> int:
    """Return VASP's band-group count ``NPAR`` for the INCAR's explicit tags on ``ranks`` ranks."""

    # VASP takes NPAR over NCORE when both are set.
    if "NPAR" in incar:
        return max(1, int(float(incar["NPAR"])))
    split = int(float(incar.get("KPAR", "1"))) * int(float(incar.get("NCORE", "1")))
    return max(1, ranks // max(1, split))


def _left_alone(incar: Mapping[str, str], ranks: int) -> str | None:
    """Return why a run gets no planned tags, or ``None`` when it is planned."""

    if ranks == 1:
        return "1 rank"
    for tag in ("LEPSILON", "LCALCEPS", "LHFCALC"):
        if _true(incar.get(tag)):
            return f"{tag} is true"
    if int(float(incar.get("IMAGES", "0"))) > 0:
        return f"IMAGES={incar['IMAGES']}"
    if "IBRION" in incar and int(float(incar["IBRION"])) in {7, 8}:
        return f"IBRION={incar['IBRION']} (DFPT)"
    if incar.get("ALGO", "").upper().startswith(("GW", "BSE", "ACFDT", "RPA")):
        return f"ALGO={incar['ALGO']}"
    return None


def plan_parallelization(
    directory: str | os.PathLike[str] = ".",
    *,
    ranks: int,
    ranks_per_node: int | None = None,
    nbands_rounding: str = "off",
) -> ParallelPlan:
    """Plan ``KPAR`` and ``NCORE`` for the VASP inputs in ``directory`` on ``ranks`` ranks.

    An INCAR that sets ``NCORE``, ``NPAR`` or ``KPAR`` is respected and gets
    nothing, with a warning when its explicit ``NBANDS`` would be rounded above
    ``2 NELECT`` (custodian's unrecoverable ``auto_nbands`` case; with
    ``nbands_rounding="auto"`` the rounding step warns instead), or when the
    ranks do not split evenly into ``KPAR`` groups by ``NPAR`` (or ``NCORE``). DFPT
    (``LEPSILON``, ``LCALCEPS``, ``IBRION = 7, 8``), hybrid (``LHFCALC``),
    GW/BSE/ACFDT/RPA and nudged-elastic-band (``IMAGES``) runs, and a single
    rank, get nothing. Otherwise ``KPAR`` is the largest divisor of ``ranks``
    not above a lower bound on the irreducible k-points (for a mesh, its orbit
    count under the lattice's point group, counted up to ``ranks``) that still
    leaves every k-point group ``ceil(NIONS / 2)`` ranks (``LELF`` and
    ``LOPTICS`` force ``KPAR = 1``), and ``NCORE`` the divisor of the group
    size (and of ``ranks_per_node`` when known) closest to 8. With
    ``nbands_rounding="auto"``, which the launch wrapper passes unless
    ``vasp.nbands_rounding`` is ``off`` because it then rounds ``NBANDS``
    itself, that candidate is taken whenever an explicit ``NBANDS``
    rounded to its ``NPAR`` stays at most ``2 NELECT``; otherwise, and always
    with ``"off"``, ``NCORE`` is the candidate closest to 8 that keeps an
    explicit ``NBANDS`` unchanged. For VASP's default band count it is the
    candidate that keeps that count either unchanged by rounding or, rounded, at
    most ``2 NELECT``. VASP rounds its default silently (custodian only refuses a
    changed INCAR ``NBANDS``), so that rule only keeps the default from being
    inflated. In ``"auto"`` mode an ``NBANDS`` the wrapper wrote counts as the
    base it was rounded from. A planned value of ``1`` is VASP's default and is
    not written.

    :param directory: Read ``INCAR``, ``POSCAR``, ``POTCAR`` and ``KPOINTS`` from this directory.
    :param ranks: Plan for this many MPI ranks.
    :param ranks_per_node: Keep ``NCORE`` a divisor of this per-node rank count when known.
    :param nbands_rounding: ``"auto"`` when ``NBANDS`` is rounded to ``NPAR`` after planning, ``"off"`` (the
        default, for a caller that leaves ``NBANDS`` alone) when not.
    :return: The plan.
    :raises OSError: If a needed input file cannot be read.
    :raises ValueError: If a needed input is malformed, ``ranks`` is not positive, or
        ``nbands_rounding`` is neither ``"auto"`` nor ``"off"``.
    """

    if ranks < 1:
        raise ValueError("ranks must be positive")
    if nbands_rounding not in ("auto", "off"):
        raise ValueError(f"nbands_rounding must be 'auto' or 'off', not {nbands_rounding!r}")
    root = Path(directory)
    text = (root / "INCAR").read_text(encoding="utf-8")
    incar = read_incar(root / "INCAR")
    if nbands_rounding == "auto":
        requested = _requested_nbands(text, incar)
    else:
        requested = int(float(incar["NBANDS"])) if "NBANDS" in incar else None
    explicit = [tag for tag in ("NPAR", "NCORE", "KPAR") if tag in incar]
    if explicit:
        settings = " ".join(f"{tag}={incar[tag]}" for tag in explicit)
        warnings: list[str] = []
        # With rounding on, the rounding step warns about this itself, except in a run it leaves to VASP.
        if requested is not None and (nbands_rounding == "off" or _left_alone(incar, ranks) is not None):
            nelect = _nelect(root, incar, read_poscar_header(root / "POSCAR"))
            npar = _band_groups(incar, ranks)
            rounded = _rounded(requested, npar)
            if rounded != requested and rounded > 2 * nelect:
                warnings.append(
                    f"NBANDS={requested} rounds up to {rounded} under NPAR={npar} from the explicit {settings}, above "
                    f"2 x NELECT = {2 * nelect:g}; custodian's VaspErrorHandler refuses that run as unrecoverable"
                )
        kpar = int(float(incar.get("KPAR", "1")))
        # NPAR, when set, is the band-group count VASP uses; otherwise NCORE is the group size.
        divider = "NPAR" if "NPAR" in incar else "NCORE"
        size = int(float(incar.get(divider, "1")))
        if kpar < 1 or size < 1 or ranks % kpar or (ranks // kpar) % size:
            warnings.append(f"{ranks} ranks do not split evenly into KPAR={kpar} groups by {divider}={size}")
        return ParallelPlan(ranks, None, None, f"{ranks} ranks, INCAR sets {settings}", tuple(warnings))
    reason = _left_alone(incar, ranks)
    if reason is not None:
        return ParallelPlan(ranks, None, None, f"{ranks} ranks, left to VASP: {reason}")

    header = read_poscar_header(root / "POSCAR")
    nions = sum(header.counts)
    nelect = _nelect(root, incar, header)
    if _true(incar.get("LELF")) or _true(incar.get("LOPTICS")):
        n_k, source = 1, "LELF or LOPTICS, KPAR=1"
    else:
        n_k, source = _kpoint_bound(root, incar, header, limit=ranks)
    # Each k-point group keeps enough ranks for the bands and plane waves of a large cell.
    least = max(1, math.ceil(nions / 2))
    kpar = max(item for item in _divisors(ranks) if item == 1 or (item <= n_k and ranks // item >= least))
    group = ranks // kpar
    candidates = [item for item in _divisors(group) if ranks_per_node is None or ranks_per_node % item == 0]
    nbands = _default_nbands(nelect, nions, incar) if requested is None else requested
    start = max(item for item in candidates if item <= _PREFERRED_NCORE)

    def fits(item: int) -> bool:
        # An unchanged count cannot trip custodian's check, whatever 2 NELECT is.
        rounded = _rounded(nbands, group // item)
        return rounded == nbands or rounded <= 2 * nelect

    if requested is not None:
        bands = f"NBANDS={nbands} explicit"
        order = [item for item in reversed(candidates) if item <= _PREFERRED_NCORE]
        order += [item for item in candidates if item > _PREFERRED_NCORE]
        if nbands_rounding == "auto" and fits(start):
            # The wrapper rounds NBANDS up to this NPAR afterwards, staying within 2 NELECT.
            ncore = start
        else:
            ncore = next((item for item in order if nbands % (group // item) == 0), group)
    else:
        bands = f"default NBANDS~{nbands}"
        ncore = next((item for item in candidates if item >= start and fits(item)), group)
    return ParallelPlan(
        ranks,
        kpar if kpar > 1 else None,
        ncore if ncore > 1 else None,
        f"{ranks} ranks, KPAR={kpar} NCORE={ncore} ({bands}, NELECT={nelect:g}, NIONS={nions}, "
        f">={n_k} irreducible k-points from {source})",
    )


def _round_nbands(directory: str | os.PathLike[str], *, ranks: int) -> tuple[str | None, str, tuple[str, ...]]:
    """Round the INCAR's ``NBANDS`` to its band-group count on ``ranks`` ranks, as the launch wrapper does.

    ``NPAR`` is the INCAR's explicit ``NPAR``, else ``ranks / (KPAR NCORE)`` with
    the INCAR's values (VASP's ``1`` when absent). An explicit ``NBANDS`` is
    rounded up to a multiple of it, an absent one becomes VASP's default rounded
    the same way, each only while the result stays at most ``2 NELECT``; a value
    written carries the marker comment that :func:`_requested_nbands` reads back,
    so a value written before is recomputed from its base. The cases
    :func:`plan_parallelization` leaves to VASP, and ``NPAR = 1``, round nothing,
    but still put a value written before back to its base. Bare marker lines are
    removed.

    :param directory: Read ``INCAR``, ``POSCAR`` and ``POTCAR`` from this directory.
    :param ranks: Round for this many MPI ranks.
    :return: The rewritten INCAR text, or ``None`` when it stays as it is; one line
        explaining the decision; and warnings.
    :raises OSError: If a needed input file cannot be read.
    :raises ValueError: If a needed input is malformed.
    """

    root = Path(directory)
    text = (root / "INCAR").read_text(encoding="utf-8")
    incar = read_incar(root / "INCAR")
    marker = _marker(text, incar)
    requested = _requested_nbands(text, incar)
    current = (int(float(incar["NBANDS"])) if "NBANDS" in incar else None, None if marker is None else marker[2])
    warnings: tuple[str, ...] = ()
    reason = _left_alone(incar, ranks)
    if reason is not None:
        # A value this wrapper wrote on an earlier launch goes back to what the INCAR asked for.
        context, source = f"left to VASP: {reason}", "default" if requested is None else str(requested)
        target: tuple[int | None, str | None] = (requested, None) if marker is not None else current
    else:
        npar = _band_groups(incar, ranks)
        if "NPAR" in incar:
            context = f"NPAR={npar} explicit"
        else:
            kpar, ncore = incar.get("KPAR", "1"), incar.get("NCORE", "1")
            context = f"NPAR={npar} from {ranks} ranks / (KPAR={kpar} x NCORE={ncore})"
        if npar <= 1 and marker is None:
            if _BARE_MARKER.search(text) is None:
                return None, f"{context}, nothing to round", ()
            # Only stray marker lines to remove: nothing that needs the POSCAR or POTCAR.
            target, source = current, ""
        else:
            header = read_poscar_header(root / "POSCAR")
            nelect = _nelect(root, incar, header)
            base = _default_nbands(nelect, sum(header.counts), incar) if requested is None else requested
            source = f"default {base}" if requested is None else str(base)
            rounded = _rounded(base, npar)
            if rounded == base and requested is not None:
                target = (base, None)
            elif npar > 1 and rounded <= 2 * nelect:
                target = (rounded, f"# httk: rounded from {source} for NPAR {npar}")
            else:
                target = (requested, None)
                if rounded != base:
                    limit = f"rounds up to {rounded} under {context}, above 2 x NELECT = {2 * nelect:g}"
                    if requested is None:
                        warnings = (
                            f"VASP's default NBANDS {base} {limit}; not written, VASP rounds its default itself",
                        )
                    else:
                        refusal = "custodian's VaspErrorHandler refuses that run as unrecoverable"
                        advice = (
                            f"not written: VASP rounds it itself and {refusal}; remove NBANDS to restore VASP's default"
                        )
                        warnings = (f"NBANDS={base} {limit}; {advice}",)
    before = incar.get("NBANDS", "absent")
    # Marker lines left bare (by an older INCAR rewrite, or by hand) describe nothing any more.
    cleaned = _BARE_MARKER.sub("", text)
    if target == current:
        if cleaned == text:
            return None, f"{context}, NBANDS {before} unchanged", warnings
        return cleaned, f"{context}, stray marker line removed, NBANDS {before} unchanged", warnings
    value, comment = target
    written = None if value is None else f"{value}  {comment}" if comment else value
    after = "absent (VASP's default)" if value is None else str(value)
    return (
        _updated_incar(cleaned, {"NBANDS": written}),
        f"{context}, NBANDS {before} -> {after} (from {source})",
        warnings,
    )
