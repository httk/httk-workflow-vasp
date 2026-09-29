"""The VASP remedy, cleanup and diagnostic helpers, moved from httk-workflow's Bash API tests."""

import json
import sys
from pathlib import Path

from httk.workflow.supervision import Diagnostic

from httk.codes.vasp import (
    VaspRemedyDecision,
    apply_vasp_remedy,
    clean_vasp_outputs,
    diagnose_vasp_files,
    plan_vasp_remedy,
    run_vasp,
)


def _poscar(path: Path) -> None:
    path.write_text(
        """test
1
2 0 0
0 2 0
0 0 2
Si
1
Direct
0 0 0
""",
        encoding="utf-8",
    )


def test_vasp_remedy_is_planned_then_explicitly_applied(tmp_path: Path) -> None:
    _poscar(tmp_path / "POSCAR")
    (tmp_path / "INCAR").write_text("ISPIN = 2\nISYM = 2\n", encoding="utf-8")
    (tmp_path / "KPOINTS").write_text("mesh\n0\nMonkhorst-Pack\n3 4 5\n0 0 0\n", encoding="utf-8")
    diagnostic = Diagnostic("kpoints_class", "error", "mismatch", "stdout")
    decision = plan_vasp_remedy(
        (diagnostic,),
        directory=tmp_path,
        history_path=tmp_path / ".httk-vasp" / "remedies.json",
    )
    assert not decision.give_up
    assert decision.changes == (("bump_kpoints", 1),)
    apply_vasp_remedy(decision, directory=tmp_path)
    assert (tmp_path / "KPOINTS").read_text(encoding="utf-8").splitlines()[3] == "4 5 6"
    history = json.loads((tmp_path / ".httk-vasp" / "remedies.json").read_text(encoding="utf-8"))
    assert history["attempts"]["kpoints_class"] == 1
    assert history["events"][0]["files"][0]["before_sha256"]
    assert history["events"][0]["files"][0]["after_sha256"]


def test_vasp_zpotrf_remedy_is_bounded_and_rounds_bands(tmp_path: Path) -> None:
    _poscar(tmp_path / "POSCAR")
    (tmp_path / "INCAR").write_text("NBANDS = 10\nNPAR = 4\n", encoding="utf-8")
    # Planning validates every rung against the real directory, so the KPOINTS the
    # second rung modifies has to be staged for that rung to be proposed at all.
    (tmp_path / "KPOINTS").write_text("mesh\n0\nMonkhorst-Pack\n3 3 3\n0 0 0\n", encoding="utf-8")
    decision = plan_vasp_remedy(
        (Diagnostic("zpotrf", "fatal", "factorization failed", "stdout"),),
        directory=tmp_path,
        history_path=tmp_path / ".httk-vasp" / "remedies.json",
    )
    assert not decision.give_up
    assert decision.changes == (("scale_lattice", 1.05),)
    apply_vasp_remedy(decision, directory=tmp_path)
    assert (tmp_path / "POSCAR").read_text(encoding="utf-8").splitlines()[1] == "1.05"

    second = plan_vasp_remedy(
        (Diagnostic("zpotrf", "fatal", "factorization failed", "stdout"),),
        directory=tmp_path,
        history_path=tmp_path / ".httk-vasp" / "remedies.json",
    )
    assert second.changes == (("bump_kpoints", 1),)

    bands = VaspRemedyDecision(
        "reviewed-v1",
        "zpotrf",
        2,
        (("bump_bands", 2),),
        False,
        "reviewed remedy available",
    )
    apply_vasp_remedy(bands, directory=tmp_path)
    assert "NBANDS = 12" in (tmp_path / "INCAR").read_text(encoding="utf-8")


def test_vasp_preclean_preserves_declared_outputs(tmp_path: Path) -> None:
    for name in ("OUTCAR", "WAVECAR", "keep.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    removed = clean_vasp_outputs(tmp_path, keep=("WAVECAR",))
    assert {path.name for path in removed} == {"OUTCAR"}
    assert (tmp_path / "WAVECAR").is_file()
    assert (tmp_path / "keep.txt").is_file()


def test_vasp_5_and_6_diagnostics_are_structured(tmp_path: Path) -> None:
    program = tmp_path / "fake-vasp.py"
    program.write_text(
        """from pathlib import Path
print("Fatal error: unable to match k-point")
print("ERROR FEXCF: supplied exchange-correlation table")
Path("OUTCAR").write_text("vasp.6.4\\n")
Path("OSZICAR").write_text("")
""",
        encoding="utf-8",
    )
    report = run_vasp([sys.executable, str(program)], directory=tmp_path, termination_grace=0.1)
    assert report.classification == "diagnosed_stop"
    assert {item.code for item in report.diagnostics} >= {
        "tetrahedron_kpoints",
        "fexcf",
        "incomplete_outcar",
    }
    saved = json.loads((tmp_path / "vasp-run-report.json").read_text(encoding="utf-8"))
    assert saved["format"] == "httk-vasp-run-report"

    for noun in ("information", "informations"):
        (tmp_path / "OUTCAR").write_text(
            f"General timing and accounting {noun} for this job:\n",
            encoding="utf-8",
        )
        assert "incomplete_outcar" not in {item.code for item in diagnose_vasp_files(tmp_path)}
