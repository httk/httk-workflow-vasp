"""The data-oriented VASP helpers, moved from httk-workflow's runtime-helper tests."""

import bz2
from pathlib import Path

from httk.codes.vasp import (
    assemble_potcar,
    automatic_kpoint_grid,
    contcar_to_poscar,
    last_oszicar_energy,
    read_incar,
    read_poscar_header,
    suggested_magnetic_moments,
    update_incar,
    write_automatic_kpoints,
)


def _poscar(path: Path, *, comment: str = "example") -> None:
    path.write_text(
        f"""{comment}
1
2 0 0
0 2 0
0 0 2
Si O
1 2
Direct
0 0 0
0.25 0.25 0.25
0.75 0.75 0.75
""",
        encoding="utf-8",
    )


def test_data_oriented_vasp_helpers(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    _poscar(poscar, comment="example [MAGMOM=1*2 2*0]")
    header = read_poscar_header(poscar)
    assert header.species == ("Si", "O")
    assert header.counts == (1, 2)
    assert suggested_magnetic_moments(poscar) == "1*2 2*0"
    grid = automatic_kpoint_grid(10, poscar=poscar)
    assert grid == (6, 6, 6)
    kpoints = write_automatic_kpoints(grid, tmp_path / "KPOINTS")
    assert "6 6 6" in kpoints.read_text(encoding="utf-8")

    incar = tmp_path / "INCAR"
    incar.write_text("ENCUT = 400\nISPIN=2 # old\n", encoding="utf-8")
    update_incar({"encut": 520, "sigma": 0.05}, incar)
    assert read_incar(incar) == {"ISPIN": "2", "ENCUT": "520", "SIGMA": "0.05"}

    library = tmp_path / "potentials"
    (library / "Si_sv").mkdir(parents=True)
    (library / "O").mkdir()
    (library / "Si_sv" / "POTCAR").write_bytes(b"silicon\n")
    (library / "O" / "POTCAR.bz2").write_bytes(bz2.compress(b"oxygen\n"))
    assembled = assemble_potcar(library, poscar=poscar, output=tmp_path / "POTCAR")
    assert assembled.path.read_bytes() == b"silicon\noxygen\n"
    assert [item.variant for item in assembled.choices] == ["Si_sv", "O"]

    oszicar = tmp_path / "OSZICAR"
    oszicar.write_text(" 1 F= -.1 E0= -1.25E+01 d E = 0\n", encoding="utf-8")
    assert last_oszicar_energy(oszicar) == -12.5
    contcar = tmp_path / "CONTCAR"
    _poscar(contcar, comment="truncated")
    converted = contcar_to_poscar(contcar, reference=poscar, output=tmp_path / "POSCAR.new")
    assert converted.read_text(encoding="utf-8").splitlines()[0] == "example [MAGMOM=1*2 2*0]"
