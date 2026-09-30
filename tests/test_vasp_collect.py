"""The collect-hook helpers locate result files in workdirs and published data, and read them."""

import runpy
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

pytest.importorskip("httk.atomistic")

import httk.core
from httk.workflow.collecting import JobRecord

from httk.codes.vasp.collect import job_parameter, read_structure, read_total_energy, result_file

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


def _record(root: Path) -> JobRecord:
    return JobRecord(
        workspace_root=root,
        workspace_id="ws",
        job_id="12345678-1234-4234-8234-123456789abc",
        job_key="job--12345678-1234-4234-8234-123456789abc",
        job={"workflow": "vasp.relax"},
        runner_provenance=None,
        state="succeeded",
        failure=None,
        placement=PurePosixPath("jobs"),
        payload_path=PurePosixPath("jobs/job--12345678-1234-4234-8234-123456789abc"),
        workdir_path=None,
        data_path=PurePosixPath("data"),
        data_generation=1,
        provenance={},
        runner_steps=None,
        children={},
        declarations={},
    )


def _write(root: Path, *parts: str) -> None:
    directory = root.joinpath(*parts[:-1])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / parts[-1]).write_text(_POSCAR if parts[-1] == "CONTCAR" else _OUTCAR, encoding="utf-8")


def _workdir_record(root: Path) -> JobRecord:
    return replace(_record(root), workdir_path=PurePosixPath("run"), data_path=None, data_generation=None)


def test_published_data_is_read_below_the_data_prefix(tmp_path: Path) -> None:
    _write(tmp_path / "data" / "vasp", "OUTCAR")
    assert result_file(_record(tmp_path), "OUTCAR", data_prefix="vasp") == tmp_path / "data" / "vasp" / "OUTCAR"


@pytest.mark.parametrize("prefix", ("", "custom/results"))
def test_a_published_name_replaces_the_workdir_name_only_in_data(tmp_path: Path, prefix: str) -> None:
    _write(tmp_path / "data" / prefix / "static", "OUTCAR")
    _write(tmp_path / "run", "OUTCAR")
    published = replace(_record(tmp_path), workdir_path=PurePosixPath("run"))
    path = result_file(published, "OUTCAR", data_prefix=prefix, published="static/OUTCAR")
    assert path == tmp_path / "data" / prefix / "static" / "OUTCAR"
    path = result_file(_workdir_record(tmp_path), "OUTCAR", data_prefix=prefix, published="static/OUTCAR")
    assert path == tmp_path / "run" / "OUTCAR"


def test_missing_file_names_job_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"ws:12345678-1234-4234-8234-123456789abc.*OUTCAR"):
        result_file(_record(tmp_path), "OUTCAR", data_prefix="vasp")


@pytest.mark.parametrize("generation", (None, 1))
def test_transactional_job_does_not_fall_back_to_unpublished_workdir(tmp_path: Path, generation: int | None) -> None:
    _write(tmp_path / "run", "CONTCAR")
    record = replace(_record(tmp_path), workdir_path=PurePosixPath("run"), data_generation=generation)
    with pytest.raises(ValueError, match="expected published data file"):
        result_file(record, "CONTCAR", data_prefix="vasp")


@pytest.mark.parametrize("workdir", (None, PurePosixPath("run")))
def test_missing_workdir_result_names_job_identity(tmp_path: Path, workdir: PurePosixPath | None) -> None:
    record = replace(_record(tmp_path), workdir_path=workdir, data_path=None, data_generation=None)
    with pytest.raises(ValueError, match=r"ws:12345678-1234-4234-8234-123456789abc.*CONTCAR"):
        result_file(record, "CONTCAR")


def test_job_parameter_reads_strings_and_defaults_otherwise(tmp_path: Path) -> None:
    record = replace(_record(tmp_path), job={"parameters": {"data_prefix": "", "timeout": 5}})
    assert job_parameter(record, "data_prefix", "vasp") == ""
    assert job_parameter(record, "timeout", "x") == "x"
    assert job_parameter(record, "absent", "vasp") == "vasp"
    assert job_parameter(_record(tmp_path), "data_prefix", "vasp") == "vasp"


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
