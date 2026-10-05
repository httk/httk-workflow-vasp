"""The attempt's launch prefix reaches the vasp run helper."""

import sys
from pathlib import Path

import pytest

from httk.codes.vasp import run_vasp

PROGRAM = [sys.executable, "-c", "pass"]


def _argv(tmp_path: Path, *arguments: str, **keywords: object) -> tuple[str, ...]:
    return run_vasp(list(arguments) or PROGRAM, directory=tmp_path, **keywords).process.argv  # type: ignore[arg-type]


def test_the_prefix_is_prepended(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTK_WORKFLOW_LAUNCH", "env 'A=b c'")
    assert _argv(tmp_path)[:3] == ("env", "A=b c", sys.executable)


def test_launch_false_runs_the_command_as_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTK_WORKFLOW_LAUNCH", "env A=b")
    assert _argv(tmp_path, launch=False)[0] == sys.executable


def test_no_prefix_changes_nothing(tmp_path: Path) -> None:
    assert _argv(tmp_path)[0] == sys.executable


def test_a_launcher_is_refused_when_a_prefix_applies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTK_WORKFLOW_LAUNCH", "env A=b")
    with pytest.raises(ValueError, match="manager.launch_template"):
        run_vasp(["srun", "-n", "4", "prog"], directory=tmp_path)
