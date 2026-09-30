"""The recognized-calculation collectors of the VASP code, run through the walker."""

import bz2
import gzip
import runpy
import shutil
from pathlib import Path

import pytest

pytest.importorskip("httk.atomistic")

from httk.core.register import known_collectors
from httk.core.storage import content_id
from httk.workflow import claims, collect_tree

_POSCAR = "silicon\n1.0\n2.0 0.0 0.0\n0.0 2.0 0.0\n0.0 0.0 2.0\nSi\n2\nDirect\n0.0 0.0 0.0\n0.5 0.5 0.5\n"
_OUTCAR = runpy.run_path(str(Path(__file__).with_name("mock_vasp.py")))["_OUTCAR"]
_INCAR = "IBRION = 2\nNSW = 99\n"
_ENERGY = -10.5


def _calc(directory: Path, *, nsw: int = 99, ibrion: int = 2, incar: str = _INCAR, contcar: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    outcar = _OUTCAR.replace("NSW    =     99", f"NSW    ={nsw:7d}").replace("IBRION =      2", f"IBRION ={ibrion:7d}")
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


def test_an_unfinished_relaxation_degrades(tmp_path: Path) -> None:
    calc = _calc(tmp_path / "a", contcar=False)
    (calc / "OUTCAR").write_text(_OUTCAR.split("   FREE ENERGIE")[0], encoding="utf-8")
    (outcome,) = claims(tmp_path)
    assert outcome.kind == "claimed"
    (item,) = collect_tree(tmp_path)
    assert item.missing_collector is not None


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
