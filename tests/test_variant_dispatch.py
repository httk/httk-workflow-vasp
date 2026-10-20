"""The VASP variant dispatcher selects and execs the binary the inputs need."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from httk.codes.vasp.launch import select_vasp_command

STD, NCL, GAM = ["vasp_std"], ["vasp_ncl"], ["vasp_gam"]


def _select(tmp_path: Path, incar: str = "ENCUT = 520\n", kpoints: str | None = None, **keywords: object) -> list[str]:
    (tmp_path / "INCAR").write_text(incar, encoding="utf-8")
    if kpoints is not None:
        (tmp_path / "KPOINTS").write_text(kpoints, encoding="utf-8")
    options: dict[str, object] = {"ncl": NCL, "gamma": GAM} | keywords
    return select_vasp_command(tmp_path, standard=STD, **options)  # type: ignore[arg-type]


def _mesh(style: str, mesh: str, shift: str | None = None) -> str:
    return f"mesh\n0\n{style}\n{mesh}\n" + ("" if shift is None else f"{shift}\n")


def test_plain_is_standard(tmp_path: Path) -> None:
    assert _select(tmp_path, kpoints=_mesh("Gamma", "4 4 4")) == STD


@pytest.mark.parametrize("incar", ["LSORBIT = .TRUE.\n", "LNONCOLLINEAR = T\n", "ISPIN = 2; LSORBIT = .true. ! soc\n"])
def test_non_collinear_is_ncl(tmp_path: Path, incar: str) -> None:
    assert _select(tmp_path, incar, kpoints=_mesh("Gamma", "1 1 1")) == NCL


def test_false_lsorbit_is_standard(tmp_path: Path) -> None:
    assert _select(tmp_path, "LSORBIT = .FALSE.\n") == STD


def test_non_collinear_without_ncl_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="vasp.ncl_command"):
        _select(tmp_path, "LSORBIT = .TRUE.\n", ncl=None)


@pytest.mark.parametrize(
    "kpoints",
    [_mesh("Gamma", "1 1 1"), _mesh("Monkhorst-Pack", "1 1 1", "0 0 0"), _mesh("gamma", "1 1 1", "0. 0. 0.")],
)
def test_single_point_mesh_is_gamma(tmp_path: Path, kpoints: str) -> None:
    assert _select(tmp_path, kpoints=kpoints) == GAM


@pytest.mark.parametrize(
    ("incar", "kpoints"),
    [
        ("ENCUT = 520\n", _mesh("Monkhorst-Pack", "1 1 1", "0.5 0 0")),
        ("LEPSILON = .TRUE.\n", _mesh("Gamma", "1 1 1")),
        ("LOPTICS = T\n", _mesh("Gamma", "1 1 1")),
        ("ENCUT = 520\n", _mesh("Gamma", "2 2 2")),
        ("ENCUT = 520\n", None),
        ("ENCUT = 520\n", "auto\n0\nAuto\n1\n"),
        ("ENCUT = 520\n", "explicit\n1\nReciprocal\n0 0 0 1\n"),
    ],
)
def test_other_runs_are_standard(tmp_path: Path, incar: str, kpoints: str | None) -> None:
    assert _select(tmp_path, incar, kpoints) == STD


def test_gamma_not_configured_is_standard(tmp_path: Path) -> None:
    assert _select(tmp_path, kpoints=_mesh("Gamma", "1 1 1"), gamma=None) == STD


def _marker_command(text: str) -> str:
    return f"{shlex.quote(sys.executable)} -c 'open(\"ran\", \"w\").write(\"{text}\")'"


def _launch(tmp_path: Path, incar: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    (tmp_path / "INCAR").write_text(incar, encoding="utf-8")
    argv = [sys.executable, "-m", "httk.codes.vasp.launch", "--standard", _marker_command("std"), *arguments]
    return subprocess.run(
        [*argv, "--directory", str(tmp_path)], cwd=tmp_path, capture_output=True, text=True, check=False
    )


def test_main_execs_the_selected_command(tmp_path: Path) -> None:
    assert _launch(tmp_path, "ENCUT = 520\n", "--ncl", _marker_command("ncl")).returncode == 0
    assert (tmp_path / "ran").read_text() == "std"
    assert _launch(tmp_path, "LSORBIT = .TRUE.\n", "--ncl", _marker_command("ncl")).returncode == 0
    assert (tmp_path / "ran").read_text() == "ncl"


def test_main_reports_a_missing_ncl_command(tmp_path: Path) -> None:
    result = _launch(tmp_path, "LNONCOLLINEAR = T\n")
    assert result.returncode == 2
    assert result.stderr.startswith("httk.codes.vasp.launch: ") and "--ncl" in result.stderr
    assert not (tmp_path / "ran").exists()


def test_main_names_a_missing_program(tmp_path: Path) -> None:
    result = _launch(tmp_path, "LSORBIT = .TRUE.\n", "--ncl", "no-such-vasp_ncl -x")
    assert result.returncode == 2
    assert "'no-such-vasp_ncl' (the --ncl command)" in result.stderr


_RECORDER = """#!{python}
import json, os, sys
with open({log!r}, "a") as stream:
    stream.write(json.dumps([sys.argv[0].rsplit("/", 1)[-1], *sys.argv[1:], os.getcwd()]) + "\\n")
{tail}
"""


def _fake_tools(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Return a calculation directory and an environment with a fake ``vasp_std`` and launch prefix."""

    tools = tmp_path / "bin"
    tools.mkdir()
    log = str(tmp_path / "exec.log")
    for name, tail in (("vasp_std", ""), ("fake-srun", "os.execvp(sys.argv[3], sys.argv[3:])")):
        (tools / name).write_text(_RECORDER.format(python=sys.executable, log=log, tail=tail), encoding="utf-8")
        (tools / name).chmod(0o755)
    calculation = tmp_path / "calc"
    calculation.mkdir()
    (calculation / "INCAR").write_text("IBRION = 2\n", encoding="utf-8")
    (calculation / "POSCAR").write_text(
        "diamond\n1.0\n0 1.78 1.78\n1.78 0 1.78\n1.78 1.78 0\nC\n2\nDirect\n0 0 0\n0.25 0.25 0.25\n", encoding="utf-8"
    )
    (calculation / "POTCAR").write_text("ZVAL = 4.0\n", encoding="utf-8")
    (calculation / "KPOINTS").write_text("mesh\n0\nGamma\n11 11 11\n", encoding="utf-8")
    environment = dict(os.environ)
    environment["PATH"] = f"{tools}{os.pathsep}{environment['PATH']}"
    environment["HTTK_WORKFLOW_LAUNCH"] = "fake-srun -n 32"
    environment.pop("HTTK_WORKFLOW_CONTEXT", None)
    return calculation, environment


def _wrap(calculation: Path, environment: dict[str, str], *arguments: str) -> subprocess.CompletedProcess[str]:
    argv = [sys.executable, "-P", "-m", "httk.codes.vasp.launch", "--standard", "vasp_std", *arguments]
    return subprocess.run(argv, cwd=calculation, env=environment, capture_output=True, text=True, check=False)


def test_wrapper_plans_once_and_execs_prefix_then_binary(tmp_path: Path) -> None:
    calculation, environment = _fake_tools(tmp_path)
    first = _wrap(calculation, environment)
    assert first.returncode == 0, first.stderr
    assert "parallel: 32 ranks, KPAR=32 NCORE=1" in first.stderr and "wrote KPAR=32\n" in first.stderr
    second = _wrap(calculation, environment)
    assert second.returncode == 0, second.stderr
    assert "INCAR sets KPAR=32" in second.stderr and "INCAR unchanged" in second.stderr
    incar = (calculation / "INCAR").read_text(encoding="utf-8")
    assert incar.count("KPAR") == 1 and "NCORE" not in incar
    records = [json.loads(line) for line in (tmp_path / "exec.log").read_text(encoding="utf-8").splitlines()]
    expected = [["fake-srun", "-n", "32", "vasp_std", str(calculation)], ["vasp_std", str(calculation)]]
    assert records == expected * 2
    assert len((calculation / "httk-vasp-launch.log").read_text(encoding="utf-8").splitlines()) == 2


def test_wrapper_parallel_off_and_ranks_override(tmp_path: Path) -> None:
    calculation, environment = _fake_tools(tmp_path)
    assert _wrap(calculation, environment, "--parallel", "off").returncode == 0
    assert (calculation / "INCAR").read_text(encoding="utf-8") == "IBRION = 2\n"
    environment["HTTK_VASP_PARALLELIZATION"] = "off"
    assert _wrap(calculation, environment).returncode == 0
    assert (calculation / "INCAR").read_text(encoding="utf-8") == "IBRION = 2\n"
    result = _wrap(calculation, environment, "--parallel", "auto", "--ranks", "4", "--no-launch")
    assert result.returncode == 0, result.stderr
    assert "4 ranks" in result.stderr and "ranks from --ranks" in result.stderr
    assert "KPAR = 4" in (calculation / "INCAR").read_text(encoding="utf-8")
    assert len((calculation / "httk-vasp-launch.log").read_text(encoding="utf-8").splitlines()) == 3


def test_wrapper_refuses_an_unknown_parallelization_setting(tmp_path: Path) -> None:
    calculation, environment = _fake_tools(tmp_path)
    environment["HTTK_VASP_PARALLELIZATION"] = "maybe"
    result = _wrap(calculation, environment)
    assert result.returncode == 2 and "vasp.parallelization must be" in result.stderr
    assert not (tmp_path / "exec.log").exists()


@pytest.mark.parametrize("value", [False, True, 1, "maybe"])
def test_main_refuses_a_non_string_parallelization_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], value: object
) -> None:
    from httk.codes.vasp.launch import main

    (tmp_path / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
    (tmp_path / "job.json").write_text(json.dumps({"parameters": {"vasp.parallelization": value}}), encoding="utf-8")
    monkeypatch.setenv("HTTK_WORKFLOW_JOB_DIR", str(tmp_path))
    assert main(["--standard", "no-such-vasp", "--directory", str(tmp_path)]) == 2
    assert "vasp.parallelization must be 'auto' or 'off'" in capsys.readouterr().err
