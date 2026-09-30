"""Register the VASP code support implemented by :mod:`httk.codes.vasp`."""

from httk.core.register import register_code, register_collector

register_code("vasp", bridge="httk.codes.vasp._bridge", bash_api="httk.codes.vasp:httk-vasp.sh")
register_collector("vasp.calculation.relax", package="httk.codes.vasp:collectors/vasp-relax")
register_collector("vasp.calculation.static", package="httk.codes.vasp:collectors/vasp-static")
