#!/usr/bin/env python3
"""A mock VASP, so the examples run on a machine that has no VASP.

**This is not VASP and computes nothing.** It reads the POSCAR in its working
directory and writes the files a finished VASP run leaves behind — OUTCAR,
OSZICAR, CONTCAR, and ``vasprun.xml`` — with numbers that are plausible enough for
the parsers of the workflows-vasp runners and meaningless as physics. The "relaxed"
structure is the input with its last atom nudged.

Use it exactly the way a deployment names the real thing:

.. code-block:: console

    httk workspace settings set --key vasp.command --value "$PWD/examples/mock_vasp.py" default
    httk workflow run

``HTTK_MOCK_VASP_FAIL_ONCE=1`` makes the first run in a directory fail with a
diagnosable ``ZPOTRF`` error instead, which is what makes the runners' remedy
ladder visible: the job applies one reviewed remedy and the rerun succeeds.

``HTTK_MOCK_VASP_IONIC_STEPS=N`` runs N ionic steps of two electronic steps each
instead (``NSW`` stays 99), sleeping ``HTTK_MOCK_VASP_STEP_SECONDS`` (default
0.1) per ionic step and appending OSZICAR lines as it goes. Like VASP it honours
a ``STOPCAR``: ``LSTOP = .TRUE.`` stops after the current ionic step and
``LABORT = .TRUE.`` after the current electronic step, each announced on stdout
and in the OUTCAR, and the run then finishes normally with exit status 0.
"""

import os
import sys
import time
from pathlib import Path

_OUTCAR = (
    " fake vasp 6.4.1\n"
    "   NELM   =     60;   NELMIN=  2; NELMDL= -5\n"
    "   NSW    =     99    number of steps for IOM\n"
    "   EDIFF  =   0.100E-03   stopping-criterion for ELM\n"
    "   EDIFFG = -.200E-01   stopping-criterion for IOM\n"
    "   IBRION =      2    ionic relax: 0-MD;1-quasi-New;2-CG\n"
    "   maximum number of plane-waves:    1234\n"
    "   FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n"
    "   free  energy   TOTEN  =        -9.20000000 eV\n"
    "   energy  without entropy=       -9.20000000  energy(sigma->0) =       -9.20000000\n"
    "   FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n"
    "   free  energy   TOTEN  =       -10.00000000 eV\n"
    "   energy  without entropy=      -10.00000000  energy(sigma->0) =      -10.00000000\n"
    "   FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n"
    "   free  energy   TOTEN  =       -10.30000000 eV\n"
    "   energy  without entropy=      -10.30000000  energy(sigma->0) =      -10.30000000\n"
    "   FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n"
    "   free  energy   TOTEN  =       -10.45000000 eV\n"
    "   energy  without entropy=      -10.45000000  energy(sigma->0) =      -10.45000000\n"
    "   FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n"
    "   free  energy   TOTEN  =       -10.50000000 eV\n"
    "   energy  without entropy=      -10.50000000  energy(sigma->0) =      -10.50000000\n"
    " General timing and accounting informations for this job:\n"
)
_OSZICAR = (
    "       N       E                     dE             d eps       ncg     rms\n"
    "DAV:   1    -0.100000000000E+02   -0.10000E+02   -0.30000E+01   128   0.500E+01\n"
    "DAV:   2    -0.105000000000E+02   -0.50000E+00   -0.10000E+00   128   0.100E+00\n"
    "   1 F= -.10500000E+02 E0= -.10500000E+02  d E =-.105000E+02\n"
)
_VASPRUN = (
    '<modeling><structure name="finalpos"><crystal>'
    '<i name="volume">      8.00000000 </i></crystal></structure></modeling>\n'
)


def _stopcar(tag: str) -> bool:
    path = Path("STOPCAR")
    return path.is_file() and f"{tag} = .TRUE." in path.read_text(encoding="utf-8")


def _stepping(steps: int, pause: float) -> None:
    """Run *steps* ionic steps, stopping early on a STOPCAR."""

    header = _OUTCAR.split("   FREE ENERGIE")[0]
    with open("OUTCAR", "w", encoding="utf-8") as outcar, open("OSZICAR", "w", encoding="utf-8") as oszicar:
        outcar.write(header)
        stop = None
        for ionic in range(1, steps + 1):
            for electronic in (1, 2):
                time.sleep(pause / 2)
                oszicar.write(f"DAV: {electronic:3d}    -0.100000000000E+02   -0.10000E+02   -0.30000E+01   128\n")
                oszicar.flush()
                if _stopcar("LABORT"):
                    stop = "hard"
                    break
            if stop is None:
                energy = -10.0 - ionic / 100
                oszicar.write(f"{ionic:4d} F= {energy:.8E} E0= {energy:.8E}  d E =-.1E-02\n")
                oszicar.flush()
                outcar.write(f"   free  energy   TOTEN  =       {energy:.8f} eV\n")
                if _stopcar("LSTOP"):
                    stop = "soft"
            if stop is not None:
                message = f"{stop} stop encountered!  aborting job ...\n"
                sys.stdout.write(message)
                outcar.write(message)
                break
        outcar.write(" General timing and accounting informations for this job:\n")


def main() -> int:
    """Write one mock VASP result, or fail once when asked to."""

    attempts = Path("mock-vasp-attempts")
    previous = int(attempts.read_text(encoding="utf-8")) if attempts.is_file() else 0
    attempts.write_text(str(previous + 1), encoding="utf-8")
    structure = Path("POSCAR").read_text(encoding="utf-8").splitlines()
    if os.environ.get("HTTK_MOCK_VASP_FAIL_ONCE") == "1" and previous == 0:
        # A recognized failure: the runners diagnose this line, apply one reviewed
        # remedy to the inputs, and ask for another attempt.
        print("LAPACK: Routine ZPOTRF failed!")
        Path("OUTCAR").write_text(" fake vasp 6.4.1\n   NELM   =     60\n   NSW    =     99\n", encoding="utf-8")
        Path("OSZICAR").write_text("DAV:   1    -0.100000000000E+02\n", encoding="utf-8")
        return 1
    if "HTTK_MOCK_VASP_IONIC_STEPS" in os.environ:
        _stepping(
            int(os.environ["HTTK_MOCK_VASP_IONIC_STEPS"]), float(os.environ.get("HTTK_MOCK_VASP_STEP_SECONDS", "0.1"))
        )
    else:
        Path("OUTCAR").write_text(_OUTCAR, encoding="utf-8")
        Path("OSZICAR").write_text(_OSZICAR, encoding="utf-8")
    relaxed = list(structure)
    relaxed[0] = "relaxed by the mock vasp"
    relaxed[-1] = "0.5100000000 0.5100000000 0.5100000000"
    Path("CONTCAR").write_text("\n".join(relaxed) + "\n", encoding="utf-8")
    Path("vasprun.xml").write_text(_VASPRUN, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
