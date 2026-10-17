"""The VASP variant dispatcher selects and execs the binary the inputs need."""

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
