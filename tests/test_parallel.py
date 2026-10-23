"""The KPAR/NCORE planner, the launch wrapper's NBANDS rounding, and its rank-count sources."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from httk.codes.vasp import VaspRemedyDecision, apply_vasp_remedy
from httk.codes.vasp.inputs import read_incar, read_poscar_header
from httk.codes.vasp.launch import _attempt_ranks, _parallel_lines, _setting, main
from httk.codes.vasp.parallel import _default_nbands, _kpoint_bound, plan_parallelization

DIAMOND = "diamond\n1.0\n0 1.78 1.78\n1.78 0 1.78\n1.78 1.78 0\nC\n2\nDirect\n0 0 0\n0.25 0.25 0.25\n"
CUBIC = "cubic\n1.0\n5 0 0\n0 5 0\n0 0 5\nC\n1\nDirect\n0 0 0\n"
SUPERCELL = "Si 216\n1.0\n16.3 0 0\n0 16.3 0\n0 0 16.3\nSi\n216\nDirect\n" + "0 0 0\n" * 216
H2 = "H2\n1.0\n10 0 0\n0 10 0\n0 0 10\nH\n2\nCartesian\n0 0 0\n0 0 0.74\n"
LITHIUM = "Li\n1.0\n3.44 0 0\n0 3.44 0\n0 0 3.44\nLi\n1\nDirect\n0 0 0\n"
LINE_MODE = "line\n20\nLine-mode\nReciprocal\n0 0 0\n0.5 0 0.5\n\n0.5 0 0.5\n0.5 0.25 0.75\n\n0.5 0.25 0.75\n0 0 0\n"


def _mesh(mesh: str) -> str:
    return f"mesh\n0\nGamma\n{mesh}\n"


def _write(tmp_path: Path, incar: str, kpoints: str | None = _mesh("11 11 11"), poscar: str = DIAMOND) -> Path:
    (tmp_path / "INCAR").write_text(incar, encoding="utf-8")
    (tmp_path / "POSCAR").write_text(poscar, encoding="utf-8")
    zval = 1 if poscar in {H2, LITHIUM} else 4
    (tmp_path / "POTCAR").write_text(f"TITEL = PAW\n   ZVAL   =    {zval}.000    mass and valenz\n", encoding="utf-8")
    if kpoints is not None:
        (tmp_path / "KPOINTS").write_text(kpoints, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("incar", "kpoints", "poscar", "ranks", "per_node", "expected"),
    [
        # Diamond non-self-consistent bands: one band group per k-point group keeps NBANDS = 39.
        ("ICHARG = 11\nNBANDS = 39\n", LINE_MODE, DIAMOND, 32, 32, (32, None)),
        ("ICHARG = 11\nNBANDS = 39\n", _mesh("21 21 21"), DIAMOND, 32, 32, (32, None)),
        # Diamond relaxation: 11x11x11 has 56 irreducible k-points (counted up to 32), so KPAR = 32.
        ("IBRION = 2\nISIF = 3\n", _mesh("11 11 11"), DIAMOND, 32, None, (32, None)),
        # A large supercell with one k-point keeps all ranks for bands, NCORE within a node.
        ("ENCUT = 520\n", _mesh("2 2 2"), SUPERCELL, 64, 32, (None, 8)),
        # LELF needs KPAR = 1; NCORE is still planned.
        ("LELF = .TRUE.\n", _mesh("11 11 11"), DIAMOND, 32, 32, (None, 8)),
        # LOPTICS: KPAR = 1, NCORE keeps the explicit 24 bands (NPAR = 4).
        ("LOPTICS = .TRUE.\nNBANDS = 24\n", _mesh("11 11 11"), DIAMOND, 32, 32, (None, 8)),
        # H2 defaults to 5 bands, above 2 NELECT = 4: only NPAR = 1 leaves them unchanged.
        ("ENCUT = 400\n", _mesh("1 1 1"), H2, 32, 32, (None, 32)),
        # One Li valence electron defaults to 4 bands, kept by NPAR = 4.
        ("ENCUT = 400\n", _mesh("1 1 1"), LITHIUM, 32, 32, (None, 8)),
        # Explicit tags are respected.
        ("NCORE = 4\n", _mesh("11 11 11"), DIAMOND, 32, 32, (None, None)),
        ("KPAR = 2\n", _mesh("11 11 11"), DIAMOND, 32, 32, (None, None)),
        # Runs whose parallelization VASP restricts are left alone.
        ("LEPSILON = .TRUE.\n", _mesh("11 11 11"), DIAMOND, 32, 32, (None, None)),
        ("LCALCEPS = T\n", _mesh("11 11 11"), DIAMOND, 32, 32, (None, None)),
        ("IMAGES = 4\n", _mesh("11 11 11"), DIAMOND, 32, 32, (None, None)),
        ("IBRION = 8\n", _mesh("11 11 11"), DIAMOND, 32, 32, (None, None)),
        ("LHFCALC = .TRUE.\n", _mesh("11 11 11"), DIAMOND, 32, 32, (None, None)),
        ("ALGO = gw0\n", _mesh("11 11 11"), DIAMOND, 32, 32, (None, None)),
        ("ENCUT = 520\n", _mesh("11 11 11"), DIAMOND, 1, None, (None, None)),
    ],
)
def test_plan_rules(
    tmp_path: Path,
    incar: str,
    kpoints: str,
    poscar: str,
    ranks: int,
    per_node: int | None,
    expected: tuple[int | None, int | None],
) -> None:
    plan = plan_parallelization(_write(tmp_path, incar, kpoints, poscar), ranks=ranks, ranks_per_node=per_node)
    assert (plan.kpar, plan.ncore) == expected, plan.reason
    assert plan.ranks == ranks and plan.warnings == ()


def test_explicit_npar_that_rounds_nbands_too_far_warns(tmp_path: Path) -> None:
    plan = plan_parallelization(_write(tmp_path, "NPAR = 32\nNBANDS = 39\n"), ranks=32, ranks_per_node=32)
    assert (plan.kpar, plan.ncore) == (None, None)
    assert len(plan.warnings) == 1 and "64" in plan.warnings[0] and "custodian" in plan.warnings[0]


def test_explicit_tags_that_do_not_split_the_ranks_warn(tmp_path: Path) -> None:
    plan = plan_parallelization(_write(tmp_path, "KPAR = 3\n"), ranks=32)
    assert (plan.kpar, plan.ncore) == (None, None)
    assert plan.warnings == ("32 ranks do not split evenly into KPAR=3 groups by NCORE=1",)
    plan = plan_parallelization(_write(tmp_path, "KPAR = 2\nNPAR = 3\n"), ranks=32)
    assert plan.warnings == ("32 ranks do not split evenly into KPAR=2 groups by NPAR=3",)
    assert plan_parallelization(_write(tmp_path, "KPAR = 2\nNPAR = 4\nNCORE = 3\n"), ranks=32).warnings == ()


# (NELECT, NIONS, INCAR) -> VASP's main.F default NBANDS, computed by hand.
_NBANDS_CASES = [
    (8.0, 2, {}, 8),
    (2.0, 2, {}, 5),
    (1.0, 1, {}, 4),
    (864.0, 216, {}, 541),
    (8.0, 2, {"ISPIN": "2"}, 9),
    (8.0, 2, {"ISPIN": "2", "MAGMOM": "2*3"}, 11),
    (8.0, 2, {"LNONCOLLINEAR": ".TRUE."}, 16),
]


@pytest.mark.parametrize(("nelect", "nions", "incar", "expected"), _NBANDS_CASES)
def test_default_nbands_is_vasps_main_f_formula(
    nelect: float, nions: int, incar: dict[str, str], expected: int
) -> None:
    assert _default_nbands(nelect, nions, incar) == expected


@pytest.mark.parametrize(("nelect", "nions", "incar", "expected"), _NBANDS_CASES)
def test_default_nbands_matches_pymatgen(nelect: float, nions: int, incar: dict[str, str], expected: int) -> None:
    sets = pytest.importorskip("pymatgen.io.vasp.sets")
    magmom = [3.0] * 2 if "MAGMOM" in incar else [1.0] * nions
    pymatgen_incar = {"ISPIN": int(incar.get("ISPIN", 1)), "MAGMOM": magmom, "LNONCOLLINEAR": "LNONCOLLINEAR" in incar}
    stub = SimpleNamespace(structure=[None] * nions, incar=pymatgen_incar, nelect=nelect)
    assert sets.VaspInputSet.estimate_nbands(stub) == expected


def test_explicit_nbands_without_a_k_point_split_keeps_its_count(tmp_path: Path) -> None:
    # A single k-point: 39 bands cannot be split evenly, so NPAR = 1 keeps them.
    plan = plan_parallelization(_write(tmp_path, "NBANDS = 39\n", _mesh("1 1 1")), ranks=32, ranks_per_node=32)
    assert (plan.kpar, plan.ncore) == (None, 32)


def test_auto_rounding_lets_ncore_follow_its_preference_within_2_nelect(tmp_path: Path) -> None:
    # 39 bands on 32 ranks: NCORE = 8 means NPAR = 4 and 40 bands, within 2 x NELECT = 200 but not 16.
    root = _write(tmp_path, "NELECT = 100\nNBANDS = 39\n", _mesh("1 1 1"))
    assert plan_parallelization(root, ranks=32, ranks_per_node=32, nbands_rounding="auto").ncore == 8
    assert plan_parallelization(root, ranks=32, ranks_per_node=32).ncore == 32
    with pytest.raises(ValueError, match="nbands_rounding must be"):
        plan_parallelization(root, ranks=32, nbands_rounding="on")


def test_incar_nelect_overrides_the_potcar(tmp_path: Path) -> None:
    plan = plan_parallelization(_write(tmp_path, "NELECT = 7\n"), ranks=32)
    assert "NELECT=7" in plan.reason


@pytest.mark.parametrize(
    ("incar", "kpoints", "poscar", "expected"),
    [
        ("", LINE_MODE, DIAMOND, 60),
        ("", LINE_MODE + "! a comment\n# another\n", DIAMOND, 60),
        ("", "explicit\n5\nReciprocal\n" + "0 0 0 1\n" * 5, DIAMOND, 5),
        ("", _mesh("4 4 4"), DIAMOND, 8),
        # Length 20 on a = 5 is a 4x4x4 Gamma mesh; KSPACING 0.2 gives 7x7x7, the default 0.5 3x3x3.
        ("", "auto\n0\nAuto\n20\n", CUBIC, 10),
        ("KSPACING = 0.2\n", None, CUBIC, 20),
        ("KSPACING = 0.2\nKGAMMA = .FALSE.\n", None, CUBIC, 20),
        ("KSPACING = 0.35\nKGAMMA = .FALSE.\n", None, CUBIC, 4),
        ("", None, CUBIC, 4),
        ("", "mesh\n0\nGamma\n0 4 4\n", CUBIC, 1),
        ("", "bad\n", CUBIC, 1),
    ],
)
def test_kpoint_bounds(tmp_path: Path, incar: str, kpoints: str | None, poscar: str, expected: int) -> None:
    root = _write(tmp_path, incar or "ENCUT = 520\n", kpoints, poscar)
    assert _kpoint_bound(root, read_incar(root / "INCAR"), read_poscar_header(root / "POSCAR"))[0] == expected


T1 = (
    "C2\n1.0\n0 1.7771847853521903 1.7771847853521903\n1.7593231784017298 0 1.7771847853521903\n"
    "1.7593231784017300 1.7771847853521903 0\nC\n2\nDirect\n0 0 0\n0.5 0.5 0.5\n"
)
DIAMOND_SKEWED = DIAMOND.replace("1.78 1.78 0\n", "1.78 5.34 3.56\n")
CUBIC_SKEWED = CUBIC.replace("0 5 0\n", "5 5 0\n")
S1 = (
    "C2\n1.0\n0 1.7416410896451464 1.7768293128443136\n1.7771847853521903 -0.0355436957070438 1.7768293128443136\n"
    "1.7771847853521905 1.7771847853521903 0\nC\n2\nDirect\n0 0 0\n0.5 0.5 0.5\n"
)


# Irreducible k-points under the lattice point group: VASP's NKPTS for the diamond, T1 (body-centred
# tetragonal, 16 operations) and S1 (8 operations) cells; the 216-atom cubic cell's 4 is below the 6
# VASP found for the displaced crystal, as a lower bound may be.
@pytest.mark.parametrize(
    ("kpoints", "poscar", "expected"),
    [
        (_mesh("11 11 11"), DIAMOND, 56),
        (_mesh("13 13 13"), DIAMOND, 84),
        (_mesh("4 4 4"), DIAMOND, 8),
        (_mesh("15 15 15"), T1, 288),
        (_mesh("15 15 15"), S1, 512),
        (_mesh("2 2 2"), SUPERCELL, 4),
        (_mesh("4 4 4"), CUBIC, 10),
        (_mesh("3 3 3"), CUBIC, 4),
        (_mesh("7 7 7"), CUBIC, 20),
        ("mp\n0\nMonkhorst-Pack\n4 4 4\n", CUBIC, 4),
        # Non-reduced bases of the same lattices (a3 + 2 a1 for diamond, a2 + a1 for cubic) count the same.
        (_mesh("11 11 11"), DIAMOND_SKEWED, 56),
        (_mesh("4 4 4"), DIAMOND_SKEWED, 8),
        (_mesh("4 4 4"), CUBIC_SKEWED, 10),
        # A non-uniform mesh counts full-group orbits that meet it, whether or not an operation keeps the mesh.
        (_mesh("2 4 4"), DIAMOND, 8),
        # The transposed action (k -> N^T k) gives 2 and 10 here.
        (_mesh("1 2 2"), DIAMOND, 3),
        ("mp\n0\nMonkhorst-Pack\n2 4 4\n", T1, 7),
        # A shifted mesh falls back to its size over 48; a zero shift is no shift.
        (_mesh("4 4 4") + "0.5 0.5 0.5\n", CUBIC, 2),
        (_mesh("4 4 4") + "0 0 0\n", CUBIC, 10),
    ],
)
def test_mesh_orbits(tmp_path: Path, kpoints: str, poscar: str, expected: int) -> None:
    root = _write(tmp_path, "ENCUT = 520\n", kpoints, poscar)
    assert _kpoint_bound(root, read_incar(root / "INCAR"), read_poscar_header(root / "POSCAR"))[0] == expected


def test_mesh_count_stops_at_the_limit(tmp_path: Path) -> None:
    root = _write(tmp_path, "ENCUT = 520\n")
    incar, header = read_incar(root / "INCAR"), read_poscar_header(root / "POSCAR")
    assert _kpoint_bound(root, incar, header, limit=5)[0] == 5
    assert _kpoint_bound(root, incar, header, limit=None)[0] == 56


PREFIX = ["env", "SLURM_HOSTFILE=f", "srun", "--ntasks=24", "--cpus-per-task=1"]
SRUN = ["srun", "--distribution=arbitrary"]
TWO_NODES = {"binding": {"nodes": [{"host": "a", "procs": 32}, {"host": "b", "procs": 16}]}}
ONE_NODE = {"binding": {"nodes": [{"host": "a", "procs": 32, "gpus": 0}]}}


@pytest.mark.parametrize(
    ("prefix", "command", "context", "expected"),
    [
        (SRUN, ["vasp_std"], TWO_NODES, (48, 16)),
        (SRUN, ["vasp_std"], {"resources": {"procs": 12}}, (12, None)),
        # The prefix's own total wins over the binding and the resources.
        (PREFIX, ["vasp_std"], TWO_NODES, (24, None)),
        (["env", "srun", "--ntasks=48"], ["vasp_std"], TWO_NODES, (48, 16)),
        (PREFIX, ["vasp_std"], {"resources": {"procs": 12}}, (24, None)),
        (PREFIX, ["vasp_std"], {}, (24, None)),
        (["mpirun", "-np", "8"], ["vasp_std"], ONE_NODE, (8, None)),
        (["mpirun", "-c", "4"], ["vasp_std"], {}, (4, None)),
        (["srun", "--ntasks-per-node", "8"], ["vasp_std"], TWO_NODES, (16, 8)),
        # OpenMP threads, a per-node count without nodes, a thread flag alone, or GPUs: nothing planned.
        (["srun", "--ntasks-per-node=8", "--cpus-per-task=4"], ["vasp_std"], ONE_NODE, (None, None)),
        (["srun", "-n", "8", "-c", "2"], ["vasp_std"], ONE_NODE, (None, None)),
        (["env", "OMP_NUM_THREADS=2", "srun", "-n", "8"], ["vasp_std"], {}, (None, None)),
        (["srun", "--ntasks-per-node=8"], ["vasp_std"], {}, (None, None)),
        (["srun", "--cpus-per-task=1"], ["vasp_std"], ONE_NODE, (None, None)),
        (SRUN, ["vasp_std"], {"binding": {"nodes": [{"host": "a", "procs": 8, "gpus": 1}]}}, (None, None)),
        # An explicit OMP_NUM_THREADS decides the threads over cpus per task.
        (["env", "OMP_NUM_THREADS=1", "srun", "-n", "16", "-c", "2"], ["vasp_std"], {}, (16, None)),
        # A count below 1 is not a count.
        (["srun", "-n", "0"], ["vasp_std"], ONE_NODE, (None, None)),
        (["mpirun", "-np", "-4"], ["vasp_std"], ONE_NODE, (None, None)),
        ([], ["mpirun", "-n", "6", "vasp_std"], {"resources": {"procs": 12}}, (6, None)),
        ([], ["vasp_std"], {"resources": {"procs": 12}}, (None, None)),
        (["mylauncher"], ["vasp_std"], {}, (None, None)),
    ],
)
def test_rank_sources(
    monkeypatch: pytest.MonkeyPatch,
    prefix: list[str],
    command: list[str],
    context: dict[str, object],
    expected: tuple[int | None, int | None],
) -> None:
    # conftest pins OMP_NUM_THREADS=1, which would override the cpus-per-task inference.
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert _attempt_ranks(prefix, command, context)[:2] == expected


def test_omp_num_threads_leaves_the_rank_count_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    assert _attempt_ranks(SRUN, ["vasp_std"], ONE_NODE) == (None, None, "4 OpenMP threads per rank")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert _attempt_ranks(["srun", "-n", "16", "-c", "2"], ["vasp_std"], {})[:2] == (16, None)


def test_settings_resolve_parameter_then_variable_then_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HTTK_WORKFLOW_CONTEXT", json.dumps({"settings": {"vasp.parallelization": "off"}}))
    assert _setting("vasp.parallelization") == "off"
    monkeypatch.setenv("HTTK_VASP_PARALLELIZATION", "auto")
    assert _setting("vasp.parallelization") == "auto"
    (tmp_path / "job.json").write_text(json.dumps({"parameters": {"vasp.parallelization": "off"}}), encoding="utf-8")
    monkeypatch.setenv("HTTK_WORKFLOW_JOB_DIR", str(tmp_path))
    assert _setting("vasp.parallelization") == "off"
    assert _setting("vasp.ncl_command") is None


# Diamond with NELECT = 100 on one k-point: VASP's default is max(51 + 3, 60) = 60 bands, 2 x NELECT = 200.
RICH = "NELECT = 100\n"


def _round(root: Path, ranks: int, mode: str = "auto", rounding: str = "auto") -> list[str]:
    return _parallel_lines(root, mode, rounding, ranks, [], [])


def _nbands_lines(root: Path) -> list[str]:
    return [line for line in (root / "INCAR").read_text(encoding="utf-8").splitlines() if "NBANDS" in line]


def test_an_absent_nbands_becomes_the_rounded_default_and_follows_the_ranks(tmp_path: Path) -> None:
    root = _write(tmp_path, RICH, _mesh("1 1 1"))
    lines = _round(root, 64)
    assert _nbands_lines(root) == ["NBANDS = 64  # httk: rounded from default 60 for NPAR 8"]
    assert "nbands: 64 ranks from --ranks, NPAR=8 from 64 ranks / (KPAR=1 x NCORE=8)" in lines[1]
    assert "NBANDS absent -> 64 (from default 60)" in lines[1]

    # Same ranks: nothing rewritten. NCORE = 8 is explicit now, so fewer ranks mean fewer band groups.
    text = (root / "INCAR").read_text(encoding="utf-8")
    assert "NBANDS 64 unchanged" in _round(root, 64)[1]
    assert (root / "INCAR").read_text(encoding="utf-8") == text
    _round(root, 32)
    assert _nbands_lines(root) == ["NBANDS = 60  # httk: rounded from default 60 for NPAR 4"]
    _round(root, 128)
    assert _nbands_lines(root) == ["NBANDS = 64  # httk: rounded from default 60 for NPAR 16"]
    assert "httk:" not in (root / "INCAR").read_text(encoding="utf-8").replace(_nbands_lines(root)[0], "")


def test_an_explicit_nbands_is_rounded_only_when_needed_and_shrinks_back(tmp_path: Path) -> None:
    root = _write(tmp_path, RICH + "NCORE = 8\nNBANDS = 61\n", _mesh("1 1 1"))
    _round(root, 64)
    assert _nbands_lines(root) == ["NBANDS = 64  # httk: rounded from 61 for NPAR 8"]
    _round(root, 16)
    assert _nbands_lines(root) == ["NBANDS = 62  # httk: rounded from 61 for NPAR 2"]
    # One band group: a marked value goes back to its base, a marked default to absent.
    _round(root, 8)
    assert _nbands_lines(root) == ["NBANDS = 61"]
    marked = _write(tmp_path, RICH + "NCORE = 8\nNBANDS = 64  # httk: rounded from default 60 for NPAR 8\n")
    _round(marked, 8)
    assert (marked / "INCAR").read_text(encoding="utf-8") == RICH + "NCORE = 8\n"

    divisible = _write(tmp_path, RICH + "NCORE = 8\nNBANDS = 64\n", _mesh("1 1 1"))
    text = (divisible / "INCAR").read_text(encoding="utf-8")
    assert "NBANDS 64 unchanged" in _round(divisible, 64)[1]
    assert (divisible / "INCAR").read_text(encoding="utf-8") == text


def test_an_edited_marked_value_is_explicit(tmp_path: Path) -> None:
    root = _write(
        tmp_path, RICH + "NCORE = 8\nNBANDS = 66  # httk: rounded from default 60 for NPAR 8\n", _mesh("1 1 1")
    )
    _round(root, 64)
    assert _nbands_lines(root) == ["NBANDS = 72  # httk: rounded from 66 for NPAR 8"]


def test_explicit_npar_sets_the_band_groups(tmp_path: Path) -> None:
    root = _write(tmp_path, RICH + "NPAR = 4\nNBANDS = 61\n", _mesh("1 1 1"))
    lines = _round(root, 64)
    assert "INCAR sets NPAR=4" in lines[0] and "NPAR=4 explicit" in lines[1]
    assert _nbands_lines(root) == ["NBANDS = 64  # httk: rounded from 61 for NPAR 4"]


@pytest.mark.parametrize("tag", ["LHFCALC = .TRUE.", "IBRION = 7", "ALGO = GW0", "IMAGES = 3", "LEPSILON = T"])
def test_runs_left_to_vasp_keep_their_nbands_even_with_explicit_tags(tmp_path: Path, tag: str) -> None:
    for incar in (f"{RICH}{tag}\nNCORE = 4\nNBANDS = 61\n", f"{RICH}{tag}\n"):
        root = _write(tmp_path, incar, _mesh("1 1 1"))
        lines = _round(root, 64)
        assert "nbands: 64 ranks from --ranks, left to VASP" in lines[-1]
        assert (root / "INCAR").read_text(encoding="utf-8") == incar


def test_rounding_above_2_nelect_warns_and_writes_nothing(tmp_path: Path) -> None:
    # Diamond, NELECT = 8: 8 default bands, or 9 explicit, rounded to NPAR = 32 exceed 16.
    for incar, what in (("ENCUT = 520\n", "VASP's default NBANDS 8"), ("NBANDS = 9\n", "NBANDS=9")):
        root = _write(tmp_path, incar, _mesh("1 1 1"))
        lines = _round(root, 32, mode="off")
        assert (root / "INCAR").read_text(encoding="utf-8") == incar
        assert lines[-1].startswith(f"warning: {what} rounds up to 32 under NPAR=32") and "16" in lines[-1]
    # A marked value that now rounds too far goes back to what it was rounded from.
    root = _write(tmp_path, "NBANDS = 10  # httk: rounded from 9 for NPAR 2\n", _mesh("1 1 1"))
    _round(root, 32, mode="off")
    assert (root / "INCAR").read_text(encoding="utf-8") == "NBANDS = 9\n"
    root = _write(tmp_path, "NBANDS = 8  # httk: rounded from default 8 for NPAR 2\n", _mesh("1 1 1"))
    _round(root, 32, mode="off")
    assert (root / "INCAR").read_text(encoding="utf-8") == "\n"


def test_one_warning_when_explicit_tags_round_nbands_above_2_nelect(tmp_path: Path) -> None:
    # The planner's warning stays for a caller without rounding; with rounding the rounding step warns alone.
    root = _write(tmp_path, "NPAR = 32\nNBANDS = 39\n")
    lines = _round(root, 32)
    warnings = [line for line in lines if line.startswith("warning:")]
    assert len(warnings) == 1 and "remove NBANDS to restore VASP's default" in warnings[0]
    assert len(plan_parallelization(root, ranks=32).warnings) == 1


def test_a_run_left_to_vasp_keeps_the_planners_2_nelect_warning(tmp_path: Path) -> None:
    # The rounding step leaves this run alone, so the planner's warning is the one left.
    root = _write(tmp_path, "LHFCALC = .TRUE.\nNCORE = 1\nNBANDS = 9\n")
    lines = _round(root, 32)
    warnings = [line for line in lines if line.startswith("warning:")]
    assert len(warnings) == 1 and "NBANDS=9 rounds up to 32" in warnings[0] and "custodian" in warnings[0]


def test_a_stray_marker_line_is_removed_without_reading_the_potcar(tmp_path: Path) -> None:
    root = _write(tmp_path, "NCORE = 8\n# httk: rounded from 61 for NPAR 8\nNBANDS = 64\n")
    (root / "POTCAR").unlink()
    assert _round(root, 8, mode="off")[1].endswith("stray marker line removed, NBANDS 64 unchanged")
    assert (root / "INCAR").read_text(encoding="utf-8") == "NCORE = 8\nNBANDS = 64\n"


def test_band_groups_divide_out_explicit_and_planned_kpar(tmp_path: Path) -> None:
    root = _write(tmp_path, RICH + "KPAR = 2 ; NCORE = 4\n", _mesh("1 1 1"))
    assert "NPAR=8 from 64 ranks / (KPAR=2 x NCORE=4)" in _round(root, 64)[1]
    expected = RICH + "KPAR = 2 ; NCORE = 4\nNBANDS = 64  # httk: rounded from default 60 for NPAR 8\n"
    assert (root / "INCAR").read_text(encoding="utf-8") == expected
    # Diamond 4x4x4 has 8 irreducible k-points: KPAR = 8 on 128 ranks, NCORE = 8, so NPAR = 2.
    root = _write(tmp_path, RICH + "NBANDS = 61\n", _mesh("4 4 4"))
    lines = _round(root, 128)
    assert "KPAR=8 NCORE=8" in lines[0] and "NPAR=2 from 128 ranks" in lines[1]
    expected = RICH + "KPAR = 8\nNCORE = 8\nNBANDS = 62  # httk: rounded from 61 for NPAR 2\n"
    assert (root / "INCAR").read_text(encoding="utf-8") == expected


@pytest.mark.parametrize(("tag", "ranks"), [("", 1), ("IBRION = 8\n", 64)])
def test_a_run_left_to_vasp_gets_back_what_its_incar_asked_for(tmp_path: Path, tag: str, ranks: int) -> None:
    for marked, restored in (
        ("NBANDS = 64  # httk: rounded from 61 for NPAR 8\n", "NBANDS = 61\n"),
        ("NBANDS = 64  # httk: rounded from default 60 for NPAR 8\n", ""),
    ):
        root = _write(tmp_path, RICH + tag + "NCORE = 8\n" + marked, _mesh("1 1 1"))
        lines = _round(root, ranks)
        assert "left to VASP" in lines[-1] and "NBANDS 64 -> " in lines[-1]
        assert (root / "INCAR").read_text(encoding="utf-8") == RICH + tag + "NCORE = 8\n" + restored


def test_a_remedy_on_a_marked_nbands_leaves_one_line_and_no_stray_marker(tmp_path: Path) -> None:
    root = _write(tmp_path, RICH + "NCORE = 8\n", _mesh("1 1 1"))
    _round(root, 64)
    apply_vasp_remedy(
        VaspRemedyDecision("reviewed-v1", "too_few_bands", 0, (("bump_bands", 2),), False, ""), directory=root
    )
    assert (root / "INCAR").read_text(encoding="utf-8") == RICH + "NCORE = 8\nNBANDS = 66\n"
    _round(root, 64)
    assert (root / "INCAR").read_text(
        encoding="utf-8"
    ) == RICH + "NCORE = 8\nNBANDS = 72  # httk: rounded from 66 for NPAR 8\n"
    # A marker left on a line of its own is dropped on the next launch.
    (root / "INCAR").write_text(RICH + "# httk: rounded from 61 for NPAR 8\nNCORE = 8\nNBANDS = 64\n", encoding="utf-8")
    _round(root, 64)
    assert (root / "INCAR").read_text(encoding="utf-8") == RICH + "NCORE = 8\nNBANDS = 64\n"


def test_rounding_works_with_parallelization_off_and_can_be_switched_off(tmp_path: Path) -> None:
    root = _write(tmp_path, RICH, _mesh("1 1 1"))
    lines = _round(root, 64, mode="off")
    assert lines[0] == "parallel: off (vasp.parallelization); INCAR unchanged"
    assert _nbands_lines(root) == ["NBANDS = 64  # httk: rounded from default 60 for NPAR 64"]
    root = _write(tmp_path, RICH, _mesh("1 1 1"))
    assert _round(root, 64, mode="off", rounding="off")[1] == "nbands: off (vasp.nbands_rounding); NBANDS unchanged"
    assert (root / "INCAR").read_text(encoding="utf-8") == RICH
    assert _parallel_lines(root, "auto", "auto", None, [], [])[1].startswith("nbands: rank count unknown")


def _main(root: Path, monkeypatch: pytest.MonkeyPatch, *arguments: str) -> int | None:
    monkeypatch.setattr(os, "execvp", lambda *_: None)
    (root / "INCAR").write_text(RICH, encoding="utf-8")
    argv = ["--standard", "vasp_std", "--directory", str(root), "--ranks", "64", "--no-launch", "--parallel", "off"]
    return main([*argv, *arguments])


def test_nbands_rounding_resolves_option_then_parameter_then_variable_then_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write(tmp_path, RICH, _mesh("1 1 1"))
    _main(root, monkeypatch)
    assert _nbands_lines(root)
    monkeypatch.setenv("HTTK_WORKFLOW_CONTEXT", json.dumps({"settings": {"vasp.nbands_rounding": "off"}}))
    _main(root, monkeypatch)
    assert not _nbands_lines(root)
    _main(root, monkeypatch, "--nbands-rounding", "auto")
    assert _nbands_lines(root)
    monkeypatch.setenv("HTTK_VASP_NBANDS_ROUNDING", "auto")
    _main(root, monkeypatch)
    assert _nbands_lines(root)
    (tmp_path / "job.json").write_text(json.dumps({"parameters": {"vasp.nbands_rounding": "off"}}), encoding="utf-8")
    monkeypatch.setenv("HTTK_WORKFLOW_JOB_DIR", str(tmp_path))
    _main(root, monkeypatch)
    assert not _nbands_lines(root)
    assert "nbands: off (vasp.nbands_rounding)" in (root / "httk-vasp-launch.log").read_text(encoding="utf-8")
    capsys.readouterr()
    (tmp_path / "job.json").write_text(json.dumps({"parameters": {"vasp.nbands_rounding": True}}), encoding="utf-8")
    assert _main(root, monkeypatch) == 2
    assert "vasp.nbands_rounding must be 'auto' or 'off'" in capsys.readouterr().err
