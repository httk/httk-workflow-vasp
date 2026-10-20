"""The attempt's launch prefix reaches the vasp run helper, through the launch wrapper by default."""

import shlex
import sys
from pathlib import Path

import pytest

from httk.codes.vasp import run_vasp

PROGRAM = [sys.executable, "-c", "pass"]
MOCK_VASP = Path(__file__).with_name("mock_vasp.py")


def _argv(tmp_path: Path, *arguments: str, **keywords: object) -> tuple[str, ...]:
    return run_vasp(list(arguments) or PROGRAM, directory=tmp_path, **keywords).process.argv  # type: ignore[arg-type]


def test_the_prefix_is_prepended(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTK_WORKFLOW_LAUNCH", "env 'A=b c'")
    assert _argv(tmp_path, dispatch=False)[:3] == ("env", "A=b c", sys.executable)


def test_launch_false_runs_the_command_as_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTK_WORKFLOW_LAUNCH", "env A=b")
    assert _argv(tmp_path, launch=False, dispatch=False)[0] == sys.executable


def test_no_prefix_changes_nothing(tmp_path: Path) -> None:
    assert _argv(tmp_path, dispatch=False)[0] == sys.executable


@pytest.mark.parametrize("dispatch", [True, False])
def test_a_launcher_is_refused_when_a_prefix_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dispatch: bool
) -> None:
    monkeypatch.setenv("HTTK_WORKFLOW_LAUNCH", "env A=b")
    with pytest.raises(ValueError, match="manager.launch_template"):
        run_vasp(["srun", "-n", "4", "prog"], directory=tmp_path, dispatch=dispatch)


def _inputs(tmp_path: Path) -> None:
    (tmp_path / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
    (tmp_path / "POSCAR").write_text(
        "diamond\n1.0\n0 1.78 1.78\n1.78 0 1.78\n1.78 1.78 0\nC\n2\nDirect\n0 0 0\n0.25 0.25 0.25\n", encoding="utf-8"
    )
    (tmp_path / "POTCAR").write_text("ZVAL = 4.0\n", encoding="utf-8")
    (tmp_path / "KPOINTS").write_text("mesh\n0\nGamma\n11 11 11\n", encoding="utf-8")


def test_run_vasp_goes_through_the_wrapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _inputs(tmp_path)
    monkeypatch.setenv("HTTK_VASP_NCL_COMMAND", "vasp_ncl")
    monkeypatch.setenv("HTTK_WORKFLOW_LAUNCH", "env A=b")
    monkeypatch.setenv("HTTK_WORKFLOW_CONTEXT", '{"resources": {"procs": 32}}')
    report = run_vasp([sys.executable, str(MOCK_VASP)], directory=tmp_path)
    assert report.classification == "completed", report.diagnostics
    assert report.process.argv == (
        sys.executable,
        "-P",
        "-m",
        "httk.codes.vasp.launch",
        f"--standard={shlex.join([sys.executable, str(MOCK_VASP)])}",
        "--ncl=vasp_ncl",
    )
    incar = (tmp_path / "INCAR").read_text(encoding="utf-8")
    assert "KPAR = 32" in incar and "NCORE" not in incar
    assert "parallel: 32 ranks" in (tmp_path / "vasp.err").read_text(encoding="utf-8")


def test_launch_false_reaches_the_wrapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _inputs(tmp_path)
    monkeypatch.setenv("HTTK_WORKFLOW_LAUNCH", "no-such-launcher")
    report = run_vasp([sys.executable, str(MOCK_VASP)], directory=tmp_path, launch=False)
    assert report.process.argv[-1] == "--no-launch"
    assert report.classification == "completed", report.diagnostics
    # Without a prefix only the command's own launcher flags count, and it has none.
    assert "KPAR" not in (tmp_path / "INCAR").read_text(encoding="utf-8")


def test_without_an_ncl_setting_a_non_collinear_run_keeps_its_command(tmp_path: Path) -> None:
    _inputs(tmp_path)
    (tmp_path / "INCAR").write_text("LSORBIT = .TRUE.\n", encoding="utf-8")
    report = run_vasp([sys.executable, str(MOCK_VASP)], directory=tmp_path)
    assert report.classification == "completed", report.diagnostics
    assert report.process.argv[-1] == f"--ncl={shlex.join([sys.executable, str(MOCK_VASP)])}"


def test_a_launcher_in_a_variant_setting_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTK_WORKFLOW_LAUNCH", "env A=b")
    monkeypatch.setenv("HTTK_VASP_GAMMA_COMMAND", "mpirun vasp_gam")
    with pytest.raises(ValueError, match=r"vasp\.gamma_command setting.*manager.launch_template"):
        run_vasp(["vasp_std"], directory=tmp_path)
