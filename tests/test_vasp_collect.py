"""The collect-hook helpers read a structure and a total energy from VASP files."""

import bz2
import runpy
from pathlib import Path

import pytest

pytest.importorskip("httk.atomistic")

import httk.core

from httk.codes.vasp.collect import read_structure, read_total_energy

_POSCAR = """silicon
1.0
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0
Si
2
Direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
_OUTCAR = """ vasp.5.2.12 synthetic
   FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
   free  energy   TOTEN  =       -26.00000000 eV
   energy  without entropy=      -26.00000000  energy(sigma->0) =      -26.00000000
   FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
   free  energy   TOTEN  =       -27.00000000 eV
   energy  without entropy=      -27.00000000  energy(sigma->0) =      -27.00000000
   FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
   free  energy   TOTEN  =       -27.09328752 eV
   energy  without entropy=      -27.09328752  energy(sigma->0) =      -27.09328752
  General timing and accounting informations for this job:
"""


def _write(root: Path, *parts: str) -> None:
    directory = root.joinpath(*parts[:-1])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / parts[-1]).write_text(_POSCAR if parts[-1] == "CONTCAR" else _OUTCAR, encoding="utf-8")


def test_read_structure_and_total_energy(tmp_path: Path) -> None:
    _write(tmp_path, "CONTCAR")
    _write(tmp_path, "OUTCAR")
    assert type(read_structure(tmp_path / "CONTCAR")).__name__ == "UnitcellStructureView"
    energy = read_total_energy(tmp_path / "OUTCAR")
    assert isinstance(energy, httk.core.DataRecord)
    assert energy.value == pytest.approx(-27.09328752)


def test_an_outcar_without_final_energy_is_refused(tmp_path: Path) -> None:
    (tmp_path / "OUTCAR").write_text(" vasp.5.2.12 synthetic\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot construct a total energy"):
        read_total_energy(tmp_path / "OUTCAR")


@pytest.mark.parametrize("compressed", [False, True])
def test_an_outcar_with_energy_but_no_completion_footer_is_refused(tmp_path: Path, compressed: bool) -> None:
    data = _OUTCAR.split("General timing and accounting informations")[0]
    path = tmp_path / ("OUTCAR.bz2" if compressed else "OUTCAR")
    path.write_bytes(bz2.compress(data.encode()) if compressed else data.encode())
    with pytest.raises(ValueError, match="incomplete.*completion footer"):
        read_total_energy(path)


def test_the_mock_vasp_outcar_reports_its_last_ionic_energy(tmp_path: Path) -> None:
    """Each ionic step of the example mock repeats the header, so the final energy is the last one."""

    mock = runpy.run_path(str(Path(__file__).with_name("mock_vasp.py")))
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(mock["_OUTCAR"], encoding="utf-8")
    energy = read_total_energy(outcar)
    assert isinstance(energy, httk.core.DataRecord)
    assert energy.value == pytest.approx(-10.5)


def test_mock_vasp_outcar_reads_as_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mock writes VASP's completion footer, so the OUTCAR reader reports a finished run."""
    from httk.atomistic.integrations.vasp.io import OutcarFile

    mock = runpy.run_path(str(Path(__file__).with_name("mock_vasp.py")))
    poscar_path = tmp_path / "POSCAR"
    poscar_path.write_text(_POSCAR, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    mock_main = mock["main"]
    assert mock_main() == 0
    outcar = OutcarFile(tmp_path / "OUTCAR")
    assert outcar.completed
    assert outcar.final_energies.energy_sigma0 == "-10.50000000"
