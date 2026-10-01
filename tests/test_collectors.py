"""The recognized-calculation collectors of the VASP code, run through the walker."""

import bz2
import gzip
import logging
import re
import runpy
import shutil
from pathlib import Path

import pytest

pytest.importorskip("httk.atomistic")

from httk.core.register import known_collectors
from httk.core.storage import content_id
from httk.workflow import claims, collect_tree

from httk.codes.vasp.collect import structure_precision

_POSCAR = "silicon\n1.0\n2.0 0.0 0.0\n0.0 2.0 0.0\n0.0 0.0 2.0\nSi\n2\nDirect\n0.0 0.0 0.0\n0.5 0.5 0.5\n"
_OUTCAR = runpy.run_path(str(Path(__file__).with_name("mock_vasp.py")))["_OUTCAR"]
_INCAR = "IBRION = 2\nNSW = 99\n"
_ENERGY = -10.5


def _calc(
    directory: Path,
    *,
    nsw: int = 99,
    ibrion: int = 2,
    incar: str = _INCAR,
    contcar: bool = True,
    echo: dict[str, str] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    outcar = _OUTCAR.replace("NSW    =     99", f"NSW    ={nsw:7d}").replace("IBRION =      2", f"IBRION ={ibrion:7d}")
    for key, value in (echo or {}).items():
        outcar = re.sub(rf"(?m)^   {key} *=.*$", f"   {key:<6} = {value}", outcar)
    (directory / "OUTCAR").write_text(outcar, encoding="utf-8")
    (directory / "POSCAR").write_text(_POSCAR, encoding="utf-8")
    (directory / "INCAR").write_text(incar, encoding="utf-8")
    if contcar:
        (directory / "CONTCAR").write_text(_POSCAR.replace("0.5 0.5 0.5", "0.5 0.5 0.4"), encoding="utf-8")
    return directory


def test_both_collectors_are_registered() -> None:
    assert {"vasp.calculation.relax", "vasp.calculation.static"} <= set(known_collectors())


def test_a_relaxation_is_collected(tmp_path: Path) -> None:
    _calc(tmp_path / "a")
    (item,) = collect_tree(tmp_path)
    (outcome,) = claims(tmp_path)
    assert item.record.workspace_id == "vasp.calculation.relax"
    assert item.missing_collector is None
    assert "initial_structure" in item.inputs
    assert set(item.outputs) == {"relaxed_structure", "total_energy"}
    assert item.outputs["total_energy"].value == pytest.approx(_ENERGY)  # type: ignore[attr-defined]
    assert item.run.source_id == f"vasp.calculation.relax:{outcome.identity}"


def test_a_static_run_is_collected(tmp_path: Path) -> None:
    _calc(tmp_path / "a", nsw=0, contcar=False)
    (item,) = collect_tree(tmp_path)
    assert item.record.workspace_id == "vasp.calculation.static"
    assert "initial_structure" in item.inputs
    assert set(item.outputs) == {"total_energy"}


def test_molecular_dynamics_is_unclaimed(tmp_path: Path) -> None:
    _calc(tmp_path / "a", ibrion=0)
    outcomes = list(claims(tmp_path))
    assert outcomes
    assert all(o.kind == "unclaimed" and "molecular dynamics" in (o.reason or "") for o in outcomes)
    assert list(collect_tree(tmp_path)) == []


@pytest.mark.parametrize(("nsw", "contcar"), [(99, True), (0, False)])
@pytest.mark.parametrize("compressed", [False, True])
def test_an_outcar_without_its_completion_footer_degrades(
    tmp_path: Path, nsw: int, contcar: bool, compressed: bool
) -> None:
    calc = _calc(tmp_path / "a", nsw=nsw, contcar=contcar)
    path = calc / "OUTCAR"
    data = path.read_text(encoding="utf-8").split("General timing and accounting informations")[0]
    if compressed:
        path.unlink()
        (calc / "OUTCAR.gz").write_bytes(gzip.compress(data.encode()))
    else:
        path.write_text(data, encoding="utf-8")
    (outcome,) = claims(tmp_path)
    assert outcome.kind == "claimed"
    (item,) = collect_tree(tmp_path)
    assert item.missing_collector is not None
    assert "incomplete" in item.missing_collector
    assert item.outputs == {}


def test_an_unfinished_relaxation_without_an_energy_degrades(tmp_path: Path) -> None:
    calc = _calc(tmp_path / "a", contcar=False)
    (calc / "OUTCAR").write_text(_OUTCAR.split("   FREE ENERGIE")[0], encoding="utf-8")
    (outcome,) = claims(tmp_path)
    assert outcome.kind == "claimed"
    (item,) = collect_tree(tmp_path)
    assert item.missing_collector is not None
    assert item.outputs == {}


def test_an_incomplete_outcar_fails_fast(tmp_path: Path) -> None:
    calc = _calc(tmp_path / "a")
    path = calc / "OUTCAR"
    path.write_text(
        path.read_text(encoding="utf-8").split("General timing and accounting informations")[0], encoding="utf-8"
    )
    with pytest.raises(ValueError, match="incomplete"):
        list(collect_tree(tmp_path, fail_fast=True))


def test_compressed_files_are_collected(tmp_path: Path) -> None:
    calc = _calc(tmp_path / "a")
    for name, opener in (("OUTCAR", bz2.open), ("POSCAR", gzip.open), ("CONTCAR", bz2.open)):
        data = (calc / name).read_bytes()
        (calc / name).unlink()
        with opener(calc / (name + (".gz" if opener is gzip.open else ".bz2")), "wb") as handle:
            handle.write(data)
    (item,) = collect_tree(tmp_path)
    assert item.record.workspace_id == "vasp.calculation.relax"
    assert item.missing_collector is None
    assert item.outputs["total_energy"].value == pytest.approx(_ENERGY)  # type: ignore[attr-defined]


def test_identity_follows_content_not_location(tmp_path: Path) -> None:
    _calc(tmp_path / "a")
    (first,) = claims(tmp_path)
    (tmp_path / "a").rename(tmp_path / "b")
    (moved,) = claims(tmp_path)
    assert moved.identity == first.identity
    (tmp_path / "b" / "INCAR").write_text("IBRION = 1\n", encoding="utf-8")
    (changed,) = claims(tmp_path)
    assert changed.identity != first.identity


def test_a_copy_is_collected_once(tmp_path: Path) -> None:
    _calc(tmp_path / "a")
    shutil.copytree(tmp_path / "a", tmp_path / "b")
    assert len(list(collect_tree(tmp_path))) == 1


def test_an_unchanged_relaxation_still_collects(tmp_path: Path) -> None:
    calc = _calc(tmp_path / "a")
    (calc / "CONTCAR").write_text(_POSCAR, encoding="utf-8")
    (item,) = collect_tree(tmp_path)
    assert item.missing_collector is None
    assert content_id(item.outputs["relaxed_structure"]) == content_id(item.inputs["initial_structure"])
    assert item.outputs["total_energy"].value == pytest.approx(_ENERGY)  # type: ignore[attr-defined]


def test_structure_precision_follows_the_convergence_settings(tmp_path: Path) -> None:
    _calc(tmp_path / "relax", echo={"EDIFFG": "-0.02"})
    assert structure_precision(tmp_path / "relax") == pytest.approx(0.02)
    _calc(tmp_path / "static", nsw=0, echo={"EDIFF": "1E-06"})
    assert structure_precision(tmp_path / "static") == pytest.approx(1e-6)
    _calc(tmp_path / "zero", echo={"EDIFFG": "0", "EDIFF": "1E-05"})
    assert structure_precision(tmp_path / "zero") == pytest.approx(1e-5)
    (tmp_path / "none").mkdir()
    assert structure_precision(tmp_path / "none") == 0.001


def test_collecting_emits_no_precision_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _calc(tmp_path / "a")
    with caplog.at_level(logging.WARNING):
        (item,) = collect_tree(tmp_path)
    assert item.missing_collector is None
    assert "precision" not in caplog.text
