"""The ``vasp`` code is registered through the ``codes`` registry tier."""

import argparse
from importlib.resources import files
from pathlib import Path

import httk.core  # noqa: F401  (importing httk.core runs registry discovery)
from httk.core.register import code_support, known_codes


def test_vasp_is_a_known_code_with_its_packaged_bash_api() -> None:
    assert "vasp" in known_codes()
    support = code_support("vasp")
    assert support.bash_api_path() == Path(str(files("httk.codes.vasp").joinpath("httk-vasp.sh")))


def test_the_bridge_mounts_the_vasp_commands() -> None:
    parser = argparse.ArgumentParser()
    code_support("vasp").resolve_bridge().add_commands(parser.add_subparsers(dest="command"))
    assert parser.parse_args(["vasp-prepare"]).command == "vasp-prepare"
