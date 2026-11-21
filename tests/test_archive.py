"""End-of-job finalization: OUTCAR cleaning (v1 parity), POTCAR summaries, compression, readers."""

import bz2
import gzip
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from httk.core.datastream.compression import open_compressed
from httk.workflow.runtime_utils import read_digest_manifest

from httk.codes.vasp import (
    clean_outcar,
    clean_vasp_outputs,
    contcar_to_poscar,
    copy_output,
    default_compress_workers,
    finalize_job,
    finalize_outputs,
    find_output,
    last_oszicar_energy,
    last_vasprun_volume,
    outcar_plane_wave_count,
    outcar_potim,
    read_output_bytes,
    read_output_text,
    summarize_potcar,
)

zstd = pytest.importorskip("compression.zstd" if sys.version_info >= (3, 14) else "backports.zstd")

# The VASP_CLEAN_OUTCAR program of httk v1 (Execution/tasks/vasp/vasptools.sh), verbatim.
_V1_CLEAN_OUTCAR = r"""
    /^ *k-point *[0-9*]* *: +[0-9.-]* +[0-9.-]* +[0-9.-]* *$/ && off==0 {print "VASP_CLEAN_OUTCAR: Removed k-point occupation data"; off=1}
    /^ *k-points in units of 2pi\/SCALE and weight:/ && off==0 {print "VASP_CLEAN_OUTCAR: Removed k-point listning."; off=2}
    /^ *Following reciprocal coordinates:/ && off==0 {print "VASP_CLEAN_OUTCAR: Removed k-point specification"; off=2}
    /^ *Following cartesian coordinates:/ && off==0 {print "VASP_CLEAN_OUTCAR: Removed k-point specification"; off=2}
    /^ *k-points in reciprocal lattice and weights/ && off==0 {print "VASP_CLEAN_OUTCAR: Removed k-point specification"; off=2}
    /^ *k-point +[0-9*]* .* plane waves/ && off==0 {print "VASP_CLEAN_OUTCAR: Removed k-point plane wave data"; off=2}
    off==1 && /^ *----------------------------------- */{off=0}
    off==2 && /^ *$/{off=0}
    off==0 {print $0}
"""

_WORKFLOW_BASH_API = Path(str(files("httk.workflow").joinpath("languages/bash/httk-workflow.sh")))
_VASP_BASH_API = Path(str(files("httk.codes.vasp").joinpath("httk-vasp.sh")))
_KPOINTS = ((0.0, 0.0, 0.0), (0.25, 0.0, -0.25), (0.5, 0.5, 0.5))
_DASHES = "-" * 104


def _outcar(final_footer: bool = True) -> str:
    """Return a synthetic two-ion, spin-polarized, two-ionic-step OUTCAR that OutcarFile parses."""

    lines = [
        " vasp.6.4.1 05Apr23 (build Jan 01 2024 00:00:00) complex",
        " POTCAR:    PAW_PBE Si 05Jan2001",
        "   TITEL  = PAW_PBE Si 05Jan2001",
        "",
        " Dimension of arrays:",
        "   k-points           NKPTS =      3   k-points in BZ     NKDIM =      3   number of bands    NBANDS=      4",
        "   number of dos      NEDOS =    301   number of ions     NIONS =      2",
        "",
        "   ions per type =               2",
        "",
        " Startparameter for this run:",
        "   ENCUT  =  520.0 eV  38.22 Ry    6.18 a0",
        "   NSW    =      2    number of steps for IOM",
        "   IBRION =      2    ionic relax: 0-MD 1-quasi-New 2-CG",
        "   ISIF   =      3    stress and relaxation",
        "   POTIM  = 0.5000    time-step for ionic-motion",
        "   NELM   =     60;   NELMIN=  2; NELMDL= -5     # of ELM steps",
        "   EDIFF  = 0.1E-05   stopping-criterion for ELM",
        "   EDIFFG = -.1E-01   stopping-criterion for IOM",
        "   ISPIN  =      2    spin polarized calculation?",
        "   ISMEAR =      0;   SIGMA  =   0.05  broadening in eV -4-tet -1-fermi 0-gaus",
        "   GGA    =    PE    GGA type",
        "",
        " k-points in units of 2pi/SCALE and weight: Automatic generation of k-mesh",
        *(f"   {x:10.8f}  {y:10.8f}  {z:10.8f}       0.333" for x, y, z in _KPOINTS),
        "",
        " k-points in reciprocal lattice and weights: Automatic generation of k-mesh",
        *(f"   {x:10.8f}  {y:10.8f}  {z:10.8f}       0.333" for x, y, z in _KPOINTS),
        "",
        " Following reciprocal coordinates:",
        "            Coordinates               Weight",
        *(f"  {x:9.6f} {y:9.6f} {z:9.6f}      1.000000" for x, y, z in _KPOINTS),
        "",
        " Following cartesian coordinates:",
        "            Coordinates               Weight",
        *(f"  {x / 2:9.6f} {y / 2:9.6f} {z / 2:9.6f}      1.000000" for x, y, z in _KPOINTS),
        "",
        "",
        " k-point   1 :   0.0000 0.0000 0.0000  plane waves:     345",
        " k-point   2 :   0.2500 0.0000-0.2500  plane waves:     350",
        " k-point *** :   0.5000 0.5000 0.5000  plane waves:     352",
        "",
        " maximum and minimum number of plane-waves per node :       352      345",
        "",
        " maximum number of plane-waves:       352",
        "",
    ]
    for step, (scale, energy) in enumerate(((1.0, "-10.80000000"), (1.01, "-10.84100000")), 1):
        a = 2.715 * scale
        lines += [
            f"--------------------------------------- Iteration      {step}(   1)  ---------------------------------------",
            "",
            " ------------------------ aborting loop because EDIFF is reached ----------------------------------------",
            "",
            " magnetization (x)",
            "",
            "# of ion       s       p       d       tot",
            "------------------------------------------",
            "    1        0.010   0.020   0.000   0.030",
            "    2       -0.010   0.020   0.000   0.010",
            "--------------------------------------------------",
            "tot          0.000   0.040   0.000   0.040",
            "",
            f" E-fermi :   {5.1 + step / 100:.4f}     XC(G=0):  -8.1234     alpha+bet : -4.3210",
            "",
        ]
        for spin in (1, 2):
            lines += [f" spin component {spin}", ""]
            for index, (x, y, z) in enumerate(_KPOINTS, 1):
                lines += [
                    f" k-point {index:5d} :      {x:7.4f}   {y:7.4f}   {z:7.4f}",
                    "  band No.  band energies     occupation ",
                    *(
                        f"      {band}     {-5.0 + 2.5 * band + spin / 10:8.4f}      {1.0 if band < 3 else 0.0:7.5f}"
                        for band in range(1, 5)
                    ),
                    "",
                ]
        lines += [
            _DASHES,
            "",
            "",
            "  FORCE on cell =-STRESS in cart. coord.  units (eV):",
            "  Direction    XX          YY          ZZ          XY          YZ          ZX",
            "  ---------------------------------------------------------------------------------------",
            f"  in kB      {-1.5 * step:.5f}    {-1.5 * step:.5f}    {-1.5 * step:.5f}     0.00000     0.00000     0.00000",
            "  external pressure =       -1.50 kB  Pullay stress =        0.00 kB",
            "",
            " VOLUME and BASIS-vectors are now :",
            " -----------------------------------------------------------------------------",
            "  energy-cutoff  :      520.00",
            f"  volume of cell :       {2 * a**3:.2f}",
            "      direct lattice vectors                 reciprocal lattice vectors",
            f"     0.000000000  {a:.9f}  {a:.9f}    -0.184162063  0.184162063  0.184162063",
            f"     {a:.9f}  0.000000000  {a:.9f}     0.184162063 -0.184162063  0.184162063",
            f"     {a:.9f}  {a:.9f}  0.000000000     0.184162063  0.184162063 -0.184162063",
            "",
            " POSITION                                       TOTAL-FORCE (eV/Angst)",
            " -----------------------------------------------------------------------------------",
            "      0.00000      0.00000      0.00000         0.012000     -0.012000      0.000000",
            f"      {a / 2:.5f}      {a / 2:.5f}      {a / 2:.5f}        -0.012000      0.012000      0.000000",
            " -----------------------------------------------------------------------------------",
            "    total drift:                                0.000000      0.000000      0.000000",
            "",
            "",
            "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
            "  ---------------------------------------------------",
            f"  free  energy   TOTEN  =       {energy} eV",
            "",
            f"  energy  without entropy=      {energy}  energy(sigma->0) =      {energy}",
            "",
        ]
    if final_footer:
        lines += [
            "",
            " General timing and accounting informations for this job:",
            " ========================================================",
            "",
            "                  Total CPU time used (sec):       10.000",
        ]
    return "\n".join(lines) + "\n"


def _v1_clean(path: Path) -> bytes:
    awk = shutil.which("gawk") or shutil.which("awk")
    if awk is None:
        pytest.skip("no awk to run the v1 VASP_CLEAN_OUTCAR program")
    environment = {**os.environ, "LC_ALL": "C"}
    return subprocess.run([awk, _V1_CLEAN_OUTCAR, str(path)], env=environment, capture_output=True, check=True).stdout


@pytest.mark.parametrize("footer", [True, False])
def test_clean_outcar_is_byte_identical_to_v1_and_idempotent(tmp_path: Path, footer: bool) -> None:
    outcar = tmp_path / "OUTCAR"
    # A last line without a newline gets one, as awk prints it.
    outcar.write_text(_outcar(footer).rstrip("\n"), encoding="utf-8")
    outcar.chmod(0o640)
    expected = _v1_clean(outcar)
    assert clean_outcar(outcar) == outcar
    cleaned = outcar.read_bytes()
    assert cleaned == expected
    assert outcar.stat().st_mode & 0o777 == 0o640
    text = cleaned.decode()
    for marker in ("occupation data", "listning.", "specification", "plane wave data"):
        assert f"VASP_CLEAN_OUTCAR: Removed k-point {marker}" in text
    assert "band No." not in text and "spin component 2" not in text and "plane waves:     3" not in text
    assert "E-fermi :   5.1200" in text and "maximum number of plane-waves:       352" in text

    clean_outcar(outcar)
    assert outcar.read_bytes() == cleaned
    assert _v1_clean(outcar) == cleaned
    assert [path.name for path in tmp_path.iterdir()] == ["OUTCAR"]


def test_cleaned_outcar_parses_to_the_same_values(tmp_path: Path) -> None:
    from httk.atomistic.integrations.vasp.io import OutcarFile

    original = tmp_path / "original" / "OUTCAR"
    original.parent.mkdir()
    original.write_text(_outcar(), encoding="utf-8")
    cleaned = tmp_path / "OUTCAR"
    shutil.copyfile(original, cleaned)
    clean_outcar(cleaned)
    assert cleaned.read_bytes() != original.read_bytes()

    with OutcarFile(original) as before, OutcarFile(cleaned) as after:
        assert before.nframes == 2 and before.completed and before.issues == ()
        assert before.parameters["NKPTS"] == "3" and before.magnetization == (0.030, 0.010)
        for attribute in (
            "version_string",
            "parameters",
            "ions_per_type",
            "xc",
            "potcar_titles",
            "final_energies",
            "nframes",
            "magnetization",
            "noncollinear_magnetization",
            "completed",
            "completion_evidence",
            "issues",
        ):
            value = getattr(before, attribute)
            assert not callable(value), attribute
            assert getattr(after, attribute) == value, attribute
        assert after.stresses() == before.stresses() and len(before.stresses()) == 2
        assert list(after.frames()) == list(before.frames())
        last = after.last_frame
        assert last is not None and last.forces_floats() == ((0.012, -0.012, 0.0), (-0.012, 0.012, 0.0))
        assert last.stress_gpa_voigt() is not None and after.final_energies.free_energy == "-10.84100000"
    # E-fermi is not parsed by OutcarFile; the line survives cleaning verbatim.
    assert [line for line in cleaned.read_text().splitlines() if "E-fermi" in line] == [
        line for line in original.read_text().splitlines() if "E-fermi" in line
    ]


_POTCAR = """  PAW_PBE Si 05Jan2001
 4.00000000000000000
 parameters from PSCTR are:
   VRHFIN =Si: s2p2
   TITEL  = PAW_PBE Si 05Jan2001
   POMASS =   28.085; ZVAL   =    4.000    mass and valenz
   ENMAX  =  245.345; ENMIN  =  184.009 eV
 Description
  0.1234 secret potential body
 End of Dataset
"""


def test_summarize_potcar_replaces_the_potcar_by_summary_and_checksum(tmp_path: Path) -> None:
    potcar = tmp_path / "POTCAR"
    potcar.write_text(_POTCAR, encoding="utf-8")
    original = potcar.read_bytes()
    summary = summarize_potcar(potcar)
    assert summary == tmp_path / "POTCAR.summary"
    assert not potcar.exists()
    assert "TITEL  = PAW_PBE Si 05Jan2001" in summary.read_text() and "secret" not in summary.read_text()
    checksum = (tmp_path / "POTCAR.sha256").read_text()
    assert checksum.endswith("  POTCAR\n") and len(checksum.split()[0]) == 64

    if shutil.which("sha256sum"):
        potcar.write_bytes(original)
        verified = subprocess.run(["sha256sum", "-c", "POTCAR.sha256"], cwd=tmp_path, capture_output=True, check=False)
        assert verified.returncode == 0, verified.stdout
        potcar.unlink()

    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert summarize_potcar(potcar) == summary
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    (tmp_path / "POTCAR.summary").unlink()
    with pytest.raises(FileNotFoundError):
        summarize_potcar(potcar)


def _tree(root: Path) -> Path:
    """Build a job-like payload: ``run/`` with outputs and ``files/`` with the staged inputs."""

    run = root / "run"
    (run / "relax" / "sub").mkdir(parents=True)
    (run / ".hidden").mkdir()
    files = root / "files"
    files.mkdir()
    (run / "OUTCAR").write_text(_outcar(), encoding="utf-8")
    (run / "POTCAR").write_text(_POTCAR, encoding="utf-8")
    (run / "WAVECAR").write_bytes(bytes(range(256)) * 64)
    (run / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
    (run / "relax" / "OUTCAR").write_text(_outcar(final_footer=False), encoding="utf-8")
    (run / "relax" / "sub" / "OSZICAR").write_text("   1 F= -.10E+02 E0= -.10500000E+02  d E =-.1E+02\n")
    (run / "relax" / "CHGCAR.bz2").write_bytes(bz2.compress(b"charge"))
    (run / ".dotfile").write_text("keep\n")
    (run / ".hidden" / "OUTCAR").write_text(_outcar(), encoding="utf-8")
    (run / "link").symlink_to(run / "INCAR")
    (files / "POTCAR").write_text(_POTCAR, encoding="utf-8")
    (files / "POSCAR").write_text("Si\n", encoding="utf-8")
    (files / "OUTCAR").write_text(_outcar(), encoding="utf-8")  # a staged input: never cleaned
    return root


def _listing(root: Path) -> dict[str, bytes | str]:
    """Return every path below ``root`` with its decompressed bytes (a symlink as its root-relative target)."""

    result: dict[str, bytes | str] = {}
    for path in sorted(root.rglob("*")):
        key = str(path.relative_to(root))
        if path.is_symlink():
            result[key] = "-> " + os.path.relpath(os.readlink(path), root)
        elif path.is_file():
            with path.open("rb") as raw, open_compressed(raw, compression="extension", name=path.name) as stream:
                result[key] = stream.read()
    return result


def test_finalize_outputs_end_state_and_crash_resume(tmp_path: Path) -> None:
    root = _tree(tmp_path / "job")
    run, files = root / "run", root / "files"
    expected_outcar = _outcar().encode()
    finalize_outputs(run, files, compress_roots=(run,), workers=2)

    names = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() or path.is_symlink()}
    assert names == {
        "run/OUTCAR.zst",
        "run/POTCAR.summary.zst",
        "run/POTCAR.sha256.zst",
        "run/WAVECAR.zst",
        "run/INCAR.zst",
        "run/uncompressed.sha256",
        "run/relax/OUTCAR.zst",
        "run/relax/uncompressed.sha256",
        "run/relax/sub/OSZICAR.zst",
        "run/relax/sub/uncompressed.sha256",
        "run/relax/CHGCAR.bz2",
        "run/.dotfile",
        "run/.hidden/OUTCAR",
        "run/link",
        "files/POTCAR.summary",
        "files/POTCAR.sha256",
        "files/POSCAR",
        "files/OUTCAR",
    }
    assert (files / "OUTCAR").read_bytes() == expected_outcar
    for path in run.rglob("*.zst"):
        assert path.read_bytes().startswith(b"\x28\xb5\x2f\xfd")
    outcar = zstd.decompress((run / "OUTCAR.zst").read_bytes())
    assert b"VASP_CLEAN_OUTCAR" in outcar and outcar != expected_outcar
    assert (run / ".hidden" / "OUTCAR").read_bytes() == expected_outcar
    assert zstd.decompress((run / "WAVECAR.zst").read_bytes()) == bytes(range(256)) * 64
    assert bz2.decompress((run / "relax" / "CHGCAR.bz2").read_bytes()) == b"charge"
    assert (run / "link").is_symlink() and os.readlink(run / "link") == str(run / "INCAR")
    assert zstd.decompress((run / "INCAR.zst").read_bytes()) == b"ENCUT = 520\n"
    # The plain digest manifest lists the uncompressed content of every file compressed there: the
    # cleaned OUTCAR and the POTCAR summaries included; the pre-compressed CHGCAR.bz2 is not listed.
    manifest = read_digest_manifest(run)
    assert set(manifest) == {"OUTCAR", "POTCAR.summary", "POTCAR.sha256", "WAVECAR", "INCAR"}
    for name, digest in manifest.items():
        assert digest == hashlib.sha256(zstd.decompress((run / f"{name}.zst").read_bytes())).hexdigest()
    assert manifest["OUTCAR"] == hashlib.sha256(outcar).hexdigest()
    assert set(read_digest_manifest(run / "relax")) == {"OUTCAR"}
    manifest_bytes = (run / "uncompressed.sha256").read_bytes()
    assert manifest_bytes.decode().endswith("  WAVECAR\n")  # plain sha256sum text, not compressed
    finished = _listing(root)

    # Idempotent once done.
    finalize_outputs(run, files, compress_roots=(run,), workers=2)
    assert _listing(root) == finished
    assert (run / "uncompressed.sha256").read_bytes() == manifest_bytes

    # Crash resume: a file left beside its compressed copy, and a summarized but not removed POTCAR.
    crashed = _tree(tmp_path / "crashed")
    crun, cfiles = crashed / "run", crashed / "files"
    (crun / "INCAR.zst").write_bytes(b"partial garbage")
    (cfiles / "POTCAR.summary").write_text("stale\n")
    (crun / "OUTCAR").write_bytes(_v1_clean(crun / "OUTCAR"))  # cleaned before the crash
    finalize_outputs(crun, cfiles, compress_roots=(crun,), workers=2)
    assert _listing(crashed) == finished


def test_finalize_outputs_removes_only_its_own_stale_temporaries(tmp_path: Path) -> None:
    from httk.codes.vasp.inputs import _TEMPORARY_SUFFIX

    root = _tree(tmp_path / "job")
    run, files = root / "run", root / "files"
    # What a killed finalization leaves: a partial compressed WAVECAR and an atomic writer's temporary.
    (run / f".WAVECAR.zst.{os.getpid() + 1}.tmp").write_bytes(b"partial")
    (run / "relax" / ".CHGCAR.bz2.77.tmp").write_bytes(b"partial")
    (run / f".uncompressed.sha256.{os.getpid() + 1}.tmp").write_bytes(b"partial")
    _, writer = tempfile.mkstemp(prefix=".OUTCAR.", suffix=_TEMPORARY_SUFFIX, dir=run)
    # ... and in a root that is not an output root, as a crash in summarize_potcar on files/ leaves.
    _, summary_writer = tempfile.mkstemp(prefix=".POTCAR.summary.", suffix=_TEMPORARY_SUFFIX, dir=files)
    (files / ".POSCAR.zst.12.tmp").write_bytes(b"partial")
    kept = [
        run / ".dotfile",
        run / ".WAVECAR.zst.tmp",  # no pid: not compress_files' name
        run / ".uncompressed.sha256.tmp",  # no pid: not the manifest writer's name
        run / ".notes.lz4.12.tmp",  # not a compress_files codec
        run / ".OUTCAR.short.httk-tmp",  # not 8 tempfile characters
        run / ".hidden" / ".WAVECAR.zst.12.tmp",  # inside a dot directory
        files / ".notes",
    ]
    for path in kept[1:]:
        path.write_bytes(b"other")
    finalize_outputs(run, files, compress_roots=(run,), workers=1)
    assert not Path(writer).exists() and not Path(summary_writer).exists()
    assert not (files / ".POSCAR.zst.12.tmp").exists()
    assert not (run / f".WAVECAR.zst.{os.getpid() + 1}.tmp").exists()
    assert not (run / "relax" / ".CHGCAR.bz2.77.tmp").exists()
    assert not (run / f".uncompressed.sha256.{os.getpid() + 1}.tmp").exists()
    assert all(path.exists() for path in kept)
    assert (run / "WAVECAR.zst").exists()


def test_compression_replaces_stale_compressed_siblings(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "OUTCAR").write_text(" fresh\n")
    (run / "OUTCAR.gz").write_bytes(gzip.compress(b" stale\n"))
    (run / "OUTCAR.bz2").write_bytes(bz2.compress(b" stale\n"))
    (run / "OUTCAR.zst").write_bytes(b"partial")
    (run / "CHGCAR.bz2").write_bytes(bz2.compress(b"only copy"))
    finalize_outputs(run, workers=1)
    assert sorted(path.name for path in run.iterdir()) == ["CHGCAR.bz2", "OUTCAR.zst", "uncompressed.sha256"]
    assert find_output(run, "OUTCAR") == run / "OUTCAR.zst"
    assert read_output_text(run, "OUTCAR") == " fresh\n"


def test_clean_vasp_outputs_removes_compressed_copies(tmp_path: Path) -> None:
    for name in ("OUTCAR", "OUTCAR.zst", "OSZICAR.gz", "WAVECAR.zst", "CONTCAR.zst", "POSCAR.zst", "INCAR"):
        (tmp_path / name).write_bytes(b"x")
    removed = clean_vasp_outputs(tmp_path, keep=("WAVECAR",))
    assert {path.name for path in removed} == {"OUTCAR", "OUTCAR.zst", "OSZICAR.gz"}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["CONTCAR.zst", "INCAR", "POSCAR.zst", "WAVECAR.zst"]
    assert clean_vasp_outputs(tmp_path / "missing") == ()


def test_compressed_and_linked_potcars_are_summarized(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import hashlib
    import logging

    library = tmp_path / "library"
    library.mkdir()
    (library / "POTCAR").write_text(_POTCAR, encoding="utf-8")
    run, files, linked = tmp_path / "run", tmp_path / "files", tmp_path / "run" / "linked"
    linked.mkdir(parents=True)
    files.mkdir()
    (run / "POTCAR.gz").write_bytes(gzip.compress(_POTCAR.encode()))
    (linked / "POTCAR").symlink_to(library / "POTCAR")
    (files / "POTCAR").symlink_to(tmp_path / "nowhere")
    (run / "POTCAR_Si").write_text("per-species name: not a POTCAR\n")
    with caplog.at_level(logging.WARNING, logger="httk.codes.vasp.archive"):
        finalize_outputs(run, files, compress="none")
    digest = hashlib.sha256(_POTCAR.encode()).hexdigest()
    for directory in (run, linked):
        assert not (directory / "POTCAR").exists() and not (directory / "POTCAR").is_symlink()
        assert (directory / "POTCAR.sha256").read_text() == f"{digest}  POTCAR\n"
        assert "TITEL  = PAW_PBE Si 05Jan2001" in (directory / "POTCAR.summary").read_text()
    assert not (run / "POTCAR.gz").exists() and (run / "POTCAR_Si").exists()
    assert (library / "POTCAR").read_text(encoding="utf-8") == _POTCAR  # the link target is untouched
    assert not (files / "POTCAR").is_symlink() and not (files / "POTCAR.summary").exists()
    assert "dangling POTCAR symlink" in caplog.text


def test_finalize_outputs_options(tmp_path: Path) -> None:
    root = _tree(tmp_path / "none")
    finalize_outputs(root / "run", root / "files", compress="none", summarize_potcar=False)
    assert (root / "run" / "POTCAR").read_text() == _POTCAR and (root / "files" / "POTCAR").exists()
    assert b"VASP_CLEAN_OUTCAR" in (root / "run" / "OUTCAR").read_bytes()
    assert not list(root.rglob("*.zst")) and not list(root.rglob("uncompressed.sha256"))

    root = _tree(tmp_path / "limited")
    finalize_outputs(root / "run", root / "files", compress="bz2", clean_outcar=False, workers=1)
    assert (root / "files" / "POSCAR.bz2").exists() and (root / "files" / "POTCAR.summary.bz2").exists()
    assert b"VASP_CLEAN_OUTCAR" not in read_output_bytes(root / "run", "OUTCAR")

    root = _tree(tmp_path / "roots")
    finalize_outputs(root / "run", compress_roots=(root / "run" / "relax",), workers=1)
    assert (root / "run" / "relax" / "OUTCAR.zst").exists() and (root / "run" / "OUTCAR").exists()
    assert not (root / "run" / "POTCAR").exists() and (root / "files" / "POTCAR").exists()
    assert b"VASP_CLEAN_OUTCAR" not in (root / "run" / "OUTCAR").read_bytes()  # outside the output roots

    invalid: list[dict[str, Any]] = [{"compress": "lz4"}, {"level": 99}, {"workers": 0}]
    for options in invalid:
        root = _tree(tmp_path / f"bad{len(list(tmp_path.iterdir()))}")
        before = _listing(root)
        with pytest.raises(ValueError):
            finalize_outputs(root / "run", root / "files", **options)
        assert _listing(root) == before


def test_default_compress_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    available = len(os.sched_getaffinity(0))
    host = socket.gethostname()
    here = {"binding": {"nodes": [{"host": "elsewhere", "procs": 64}, {"host": host, "procs": 1}]}}
    assert default_compress_workers(here) == 1
    here["binding"]["nodes"][1]["procs"] = 10**6
    assert default_compress_workers(here) == available
    assert default_compress_workers({"binding": {"nodes": [{"host": "elsewhere", "procs": 64}]}}) == 1
    assert default_compress_workers({"resources": {"procs": 1}}) == 1
    assert default_compress_workers({"resources": {"procs": 10**6}}) == available
    assert default_compress_workers({}) == available

    monkeypatch.setenv("SLURMD_NODENAME", "n42")
    monkeypatch.setattr(socket, "gethostname", lambda: "login.cluster.example")
    fqdn = {
        "binding": {
            "nodes": [{"host": "n41.cluster.example", "procs": 64}, {"host": "N42.cluster.example", "procs": 1}]
        }
    }
    assert default_compress_workers(fqdn) == 1
    short = {"binding": {"nodes": [{"host": "login", "procs": 1}, {"host": "other", "procs": 64}]}}
    assert default_compress_workers(short) == 1
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0})
    assert default_compress_workers({"resources": {"procs": 8}}) == 1
    monkeypatch.setenv("HTTK_WORKFLOW_CONTEXT", json.dumps({"binding": {"nodes": [{"host": "x", "procs": 4}]}}))
    assert default_compress_workers() == 1


def _attempt(root: Path, **settings: object) -> SimpleNamespace:
    def setting(name: str, default: object = None) -> object:
        return settings.get(name, default)

    return SimpleNamespace(workdir=root / "run", payload=root, setting=setting)


@pytest.mark.parametrize("null", [None, "null"])
def test_a_null_setting_means_its_default(tmp_path: Path, null: object) -> None:
    names = ("compress", "compress_level", "compress_workers", "clean_outcar", "summarize_potcar")
    nulled, default = _tree(tmp_path / "nulled"), _tree(tmp_path / "default")
    finalize_job(_attempt(nulled, **{f"vasp.{name}": null for name in names}))  # type: ignore[arg-type]
    finalize_job(_attempt(default))  # type: ignore[arg-type]
    assert (nulled / "run" / "OUTCAR.zst").exists()
    assert _listing(nulled) == _listing(default)


def test_finalize_job_reads_its_settings(tmp_path: Path) -> None:
    root = _tree(tmp_path / "defaults")
    finalize_job(_attempt(root))  # type: ignore[arg-type]
    assert (root / "run" / "OUTCAR.zst").exists() and (root / "files" / "POTCAR.summary").exists()
    assert not (root / "files" / "POTCAR").exists() and (root / "files" / "POSCAR").exists()

    root = _tree(tmp_path / "strings")
    attempt = _attempt(
        root,
        **{
            "vasp.compress": "xz",
            "vasp.compress_level": "1",
            "vasp.compress_workers": 2,
            "vasp.clean_outcar": "No",
            "vasp.summarize_potcar": "0",
        },
    )
    finalize_job(attempt)  # type: ignore[arg-type]
    assert (root / "run" / "POTCAR.xz").exists() and (root / "files" / "POTCAR").exists()
    assert b"VASP_CLEAN_OUTCAR" not in read_output_bytes(root / "run", "OUTCAR")

    for name, value in (
        ("vasp.compress", "lz4"),
        ("vasp.compress", 3),
        ("vasp.compress_level", "high"),
        ("vasp.compress_level", 42),
        ("vasp.compress_workers", 0),
        ("vasp.compress_workers", True),
        ("vasp.clean_outcar", "maybe"),
        ("vasp.summarize_potcar", 2),
    ):
        root = _tree(tmp_path / f"bad-{name}-{value}")
        before = _listing(root)
        with pytest.raises(ValueError, match=name.replace(".", r"\.")):
            finalize_job(_attempt(root, **{name: value}))  # type: ignore[arg-type]
        assert _listing(root) == before


_BASH_FINALIZE = r"""
set -euo pipefail
source "$1"
source "$2"
# A stand-in for the attempt's setting resolution: the settings come from HTTK_VASP_* here.
httk_workflow_setting() {
    local variable="HTTK_${1^^}"
    variable="${variable//./_}"
    if [ -n "${!variable+x}" ]; then printf '%s\n' "${!variable}"
    elif [ "$#" -ge 2 ]; then printf '%s\n' "$2"
    else return 1
    fi
}
httk_vasp_finalize
"""


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {
            "vasp.compress": "gz",
            "vasp.compress_level": "1",
            "vasp.clean_outcar": "false",
            "vasp.summarize_potcar": "no",
        },
        {
            "vasp.compress": "null",
            "vasp.compress_level": "null",
            "vasp.compress_workers": "null",
            "vasp.clean_outcar": "null",
            "vasp.summarize_potcar": "null",
        },
    ],
)
def test_bash_finalize_matches_finalize_job(tmp_path: Path, settings: dict[str, str]) -> None:
    python_root = _tree(tmp_path / "python")
    finalize_job(_attempt(python_root, **settings))  # type: ignore[arg-type]

    bash_root = _tree(tmp_path / "bash")
    environment = {name: value for name, value in os.environ.items() if not name.startswith("HTTK_WORKFLOW_")}
    environment |= {
        "HTTK_WORKFLOW_PYTHON": sys.executable,
        "HTTK_WORKFLOW_WORKDIR": str(bash_root / "run"),
        "HTTK_WORKFLOW_JOB_DIR": str(bash_root),
        **{"HTTK_" + name.upper().replace(".", "_"): value for name, value in settings.items()},
    }
    completed = subprocess.run(
        ["bash", "-c", _BASH_FINALIZE, "bash", str(_WORKFLOW_BASH_API), str(_VASP_BASH_API)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert _listing(bash_root) == _listing(python_root)


def test_compression_aware_readers(tmp_path: Path) -> None:
    (tmp_path / "OSZICAR").write_text("   1 F= -.10E+02 E0= -.10500000E+02  d E =-.1E+02\n")
    (tmp_path / "vasprun.xml").write_text('<i name="volume">      8.00000000 </i>\n')
    (tmp_path / "OUTCAR").write_text(" opt step  =  0.25\n   maximum number of plane-waves:    1234\n")
    (tmp_path / "CONTCAR").write_text("lost comment\n1.0\n")
    (tmp_path / "POSCAR.ref").write_text("Si reference\n1.0\n")
    (tmp_path / "WAVECAR").write_bytes(b"\x00\x01binary")
    finalize_outputs(tmp_path, workers=1)
    assert not [path for path in tmp_path.iterdir() if path.suffix != ".zst" and path.name != "uncompressed.sha256"]

    assert last_oszicar_energy(tmp_path / "OSZICAR") == -10.5
    assert last_vasprun_volume(tmp_path / "vasprun.xml") == 8.0
    assert outcar_potim(tmp_path / "OUTCAR") == 0.25
    assert outcar_plane_wave_count(tmp_path / "OUTCAR") == 1234
    output = contcar_to_poscar(tmp_path / "CONTCAR", reference=tmp_path / "POSCAR.ref", output=tmp_path / "POSCAR")
    assert output.read_text() == "Si reference\n1.0\n"

    assert find_output(tmp_path, "OUTCAR") == tmp_path / "OUTCAR.zst"
    assert find_output(tmp_path, "POSCAR") == tmp_path / "POSCAR"
    assert find_output(tmp_path, "CHGCAR") is None
    assert read_output_text(tmp_path, "OSZICAR").startswith("   1 F=")
    assert read_output_bytes(tmp_path, "POSCAR") == b"Si reference\n1.0\n"
    (tmp_path / "seed").mkdir()
    copied = copy_output(tmp_path, "WAVECAR", tmp_path / "seed" / "WAVECAR")
    assert copied.read_bytes() == b"\x00\x01binary" and (tmp_path / "WAVECAR.zst").exists()
    with pytest.raises(FileNotFoundError):
        read_output_bytes(tmp_path, "CHGCAR")
    with pytest.raises(FileNotFoundError):
        copy_output(tmp_path, "CHGCAR", tmp_path / "CHGCAR")
    with pytest.raises(FileNotFoundError):
        last_oszicar_energy(tmp_path / "missing")


def test_an_empty_output_stays_readable(tmp_path: Path) -> None:
    (tmp_path / "vasp.err").write_bytes(b"")
    (tmp_path / "vasp.out").write_bytes(b"done\n")
    finalize_outputs(tmp_path, workers=1)
    assert find_output(tmp_path, "vasp.err") == tmp_path / "vasp.err.zst"
    assert read_output_bytes(tmp_path, "vasp.err") == b""
    assert read_output_bytes(tmp_path, "vasp.out") == b"done\n"
