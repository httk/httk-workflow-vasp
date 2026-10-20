"""The KPAR/NCORE planner and the launch wrapper's rank-count sources."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from httk.codes.vasp.inputs import read_incar, read_poscar_header
from httk.codes.vasp.launch import _attempt_ranks, _setting
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
