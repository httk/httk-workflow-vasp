import json
import sys
import time
from pathlib import Path

import pytest
from httk.workflow.codes import CheckerSpec, FollowSource, SourceEvent

from httk.codes.vasp import (
    VASP_CHECKS,
    VaspRunReport,
    parse_diagnostic_policy,
    read_checker_specs,
    run_vasp,
    soft_stop_mintime,
)
from httk.codes.vasp.reports import _soft_stop_ended_early, _SoftStop, _stop_acknowledgement

_CHECKER = """\
import json, sys
for raw in sys.stdin:
    event = json.loads(raw)
    if "MY MARKER" in event.get("line", ""):
        result = {"format": "httk-workflow-checker-result", "format_version": 2, "code": "my_check",
                  "severity": "error", "summary": "marker seen", "stop": True}
        print(json.dumps(result), flush=True)
"""


def _program(line: str, sleep: float = 0.5) -> list[str]:
    return [sys.executable, "-c", f"import time; print({line!r}, flush=True); time.sleep({sleep})"]


def _codes(report: VaspRunReport) -> dict[str, bool]:
    return {item.code: item.stop for item in report.diagnostics}


def _checker(tmp_path: Path) -> Path:
    script = tmp_path / "checker.py"
    script.write_text(_CHECKER, encoding="utf-8")
    return script


def test_a_stopping_check_ends_the_run_early(tmp_path: Path) -> None:
    started = time.monotonic()
    report = run_vasp(_program("ZPOTRF failed", sleep=30), directory=tmp_path, dispatch=False, termination_grace=2)
    assert time.monotonic() - started < 20
    assert report.classification == "diagnosed_stop"
    assert _codes(report)["zpotrf"] is True
    assert report.as_mapping()["diagnostic_policy"] == {}


def test_continue_keeps_the_diagnostic_without_stopping(tmp_path: Path) -> None:
    report = run_vasp(_program("ZPOTRF failed"), directory=tmp_path, dispatch=False, policy={"zpotrf": "continue"})
    assert report.classification != "diagnosed_stop"
    zpotrf = next(item for item in report.diagnostics if item.code == "zpotrf")
    assert (zpotrf.stop, zpotrf.severity) == (False, "fatal")
    assert report.as_mapping()["diagnostic_policy"] == {"zpotrf": "continue"}


def test_ignore_drops_the_diagnostic(tmp_path: Path) -> None:
    report = run_vasp(_program("ZPOTRF failed"), directory=tmp_path, dispatch=False, policy={"zpotrf": "ignore"})
    assert "zpotrf" not in _codes(report)
    assert report.classification != "diagnosed_stop"


def test_the_string_policy_makes_a_warning_stop(tmp_path: Path) -> None:
    line = "BRMIX: very serious problems"
    assert _codes(run_vasp(_program(line), directory=tmp_path, dispatch=False))["brmix"] is False
    report = run_vasp(_program(line), directory=tmp_path, dispatch=False, policy="brmix=stop")
    assert report.classification == "diagnosed_stop"
    assert _codes(report)["brmix"] is True


def test_an_executable_checker_stops_the_run_from_a_followed_file(tmp_path: Path) -> None:
    program = [
        sys.executable,
        "-c",
        "import time; open('ML_LOGFILE', 'w').write('ZPOTRF failed\\nMY MARKER\\n'); time.sleep(1)",
    ]
    checker = CheckerSpec(
        (sys.executable, str(_checker(tmp_path))),
        sources=(FollowSource(Path("ML_LOGFILE"), "custom"), FollowSource(Path("OUTCAR"))),
    )
    report = run_vasp(program, directory=tmp_path, dispatch=False, checkers=(checker,))
    assert report.classification == "diagnosed_stop"
    assert _codes(report)["my_check"] is True
    # The built-in checks read only VASP's own output, not a checker's source.
    assert "zpotrf" not in _codes(report)

    # The previous run's source is removed, so it cannot replay into this one.
    rerun = run_vasp([sys.executable, "-c", "pass"], directory=tmp_path, dispatch=False, checkers=(checker,))
    assert rerun.classification != "diagnosed_stop"
    assert "my_check" not in _codes(rerun)


def test_a_checker_follows_only_vasp_outputs_in_the_run_directory(tmp_path: Path) -> None:
    contcar = tmp_path / "CONTCAR"
    contcar.write_text("structure\n", encoding="utf-8")
    (tmp_path / "ML_LOGFILE").symlink_to(contcar)
    for source, problem in (("CONTCAR", "only follow VASP's own output"), ("ML_LOGFILE", "not a regular file")):
        checker = CheckerSpec((sys.executable, "-c", "pass"), sources=(FollowSource(Path(source)),))
        with pytest.raises(ValueError, match=problem):
            run_vasp(["true"], directory=tmp_path, dispatch=False, checkers=(checker,))
        assert contcar.read_text(encoding="utf-8") == "structure\n"


def test_policy_parsing_refuses_bad_entries() -> None:
    assert parse_diagnostic_policy(" brmix=stop  zpotrf=ignore ") == {"brmix": "stop", "zpotrf": "ignore"}
    assert parse_diagnostic_policy(None) == parse_diagnostic_policy("") == {}
    assert parse_diagnostic_policy({"realspace_allocation_too_large": "continue"})
    with pytest.raises(ValueError, match="live checks only"):
        parse_diagnostic_policy("electronic_nonconvergence=continue")
    with pytest.raises(ValueError, match="known codes: .*brmix"):
        parse_diagnostic_policy("brimx=stop")
    for bad in ("brmix=maybe", "brmix", "brmix=stop brmix=ignore", {"brmix": 1}, 3):
        with pytest.raises(ValueError, match="diagnostic_policy"):
            parse_diagnostic_policy(bad)
    with pytest.raises(ValueError, match="unknown code"):
        parse_diagnostic_policy("brmix=stop", checks=VASP_CHECKS[:1])


def test_an_invalid_policy_leaves_the_workdir_untouched(tmp_path: Path) -> None:
    outcar = tmp_path / "OUTCAR"
    outcar.write_text("previous run\n", encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostic_policy"):
        run_vasp(["true"], directory=tmp_path, dispatch=False, policy="zpotrf=maybe")
    assert outcar.read_text(encoding="utf-8") == "previous run\n"


def test_the_bridge_run_command_takes_a_policy_and_checkers(tmp_path: Path) -> None:
    from httk.workflow._shell_bridge import _parser

    from httk.codes.vasp._bridge import run_command

    spec = tmp_path / "checker.json"
    spec.write_text(
        json.dumps(
            {
                "format": "httk-workflow-checker-spec",
                "format_version": 2,
                "argv": [sys.executable, str(_checker(tmp_path))],
            }
        ),
        encoding="utf-8",
    )
    program = [sys.executable, "-c", "print('ZPOTRF failed'); print('MY MARKER', flush=True)"]
    arguments = ["vasp-run", "--directory", str(tmp_path), "--no-dispatch", "--policy", "zpotrf=continue"]
    assert run_command(_parser().parse_args([*arguments, "--checker", str(spec), "--", *program])) == 20
    report = json.loads((tmp_path / "vasp-run-report.json").read_text(encoding="utf-8"))
    stops = {item["code"]: item["stop"] for item in report["diagnostics"]}
    assert (stops["zpotrf"], stops["my_check"]) == (False, True)
    assert report["diagnostic_policy"] == {"zpotrf": "continue"}


def test_checker_specs_resolve_files_shipped_beside_them(tmp_path: Path) -> None:
    files = tmp_path / "files"
    files.mkdir()
    (files / "brmix.awk").write_text("/BRMIX/ { print }\n", encoding="utf-8")
    (files / "POSCAR").write_text("", encoding="utf-8")
    argv = ["awk", "-f", "./brmix.awk", "POSCAR"]
    spec = {"format": "httk-workflow-checker-spec", "format_version": 2, "argv": argv}
    (files / "brmix-checker.json").write_text(json.dumps(spec), encoding="utf-8")

    (checker,) = read_checker_specs("files/brmix-checker.json", base=tmp_path)
    assert checker.argv == ("awk", "-f", str((files / "brmix.awk").resolve()), "POSCAR")
    assert read_checker_specs(["files/brmix-checker.json"], base=tmp_path) == (checker,)
    assert read_checker_specs(None, base=tmp_path) == read_checker_specs("", base=tmp_path) == ()
    with pytest.raises(ValueError, match="missing.json"):
        read_checker_specs("files/missing.json", base=tmp_path)
    spec["argv"] = ["awk", "-f", "./absent.awk"]
    (files / "brmix-checker.json").write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="absent.awk"):
        read_checker_specs("files/brmix-checker.json", base=tmp_path)
    spec["argv"] = ["awk"]
    spec["sources"] = [{"path": "POSCAR"}]
    (files / "brmix-checker.json").write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="brmix-checker.json.*only follow VASP's own output"):
        read_checker_specs("files/brmix-checker.json", base=tmp_path)


MOCK_VASP = Path(__file__).with_name("mock_vasp.py")
_FOOTER = " General timing and accounting informations for this job:\n"


def _stepping_mock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, steps: int = 20, pause: float = 0.2, **options: object
) -> VaspRunReport:
    """Run the mock VASP's STOPCAR-honouring multi-step mode under ``run_vasp``."""

    monkeypatch.setenv("HTTK_MOCK_VASP_IONIC_STEPS", str(steps))
    monkeypatch.setenv("HTTK_MOCK_VASP_STEP_SECONDS", str(pause))
    (tmp_path / "POSCAR").write_text("mock\n1\n2 0 0\n0 2 0\n0 0 2\nSi\n1\nDirect\n0 0 0\n", encoding="utf-8")
    return run_vasp([sys.executable, str(MOCK_VASP)], directory=tmp_path, dispatch=False, **options)  # type: ignore[arg-type]


def _ionic_steps(tmp_path: Path) -> int:
    return sum(" F= " in line for line in (tmp_path / "OSZICAR").read_text(encoding="utf-8").splitlines())


def _seconds(duration: str) -> int:
    hours, minutes, seconds = (int(item) for item in duration.split(":"))
    assert minutes < 60 and seconds < 60
    return (hours * 60 + minutes) * 60 + seconds


def _finishing(
    outcar: str, oszicar: str = "", *, stdout: str = "", returncode: int = 0, sleep: float = 0.0
) -> list[str]:
    """Return a program that writes the given OUTCAR and OSZICAR, prints *stdout* and exits."""

    code = (
        "import sys, time; from pathlib import Path; "
        f"Path('OUTCAR').write_text({outcar!r}); Path('OSZICAR').write_text({oszicar!r}); "
        f"print({stdout!r}, flush=True); time.sleep({sleep}); sys.exit({returncode})"
    )
    return [sys.executable, "-c", code]


def _relaxation(nsw: int, steps: int, *extra: str) -> tuple[str, str]:
    outcar = f"   NSW    =    {nsw}    number of steps for IOM\n" + "".join(extra) + _FOOTER
    return outcar, "".join(f"   {step} F= -.1E+02 E0= -.1E+02  d E =-.1E-02\n" for step in range(1, steps + 1))


def test_a_stopcar_is_removed_before_and_after_every_run(tmp_path: Path) -> None:
    (tmp_path / "STOPCAR").write_text("LSTOP = .TRUE.\n", encoding="utf-8")
    program = [
        sys.executable,
        "-c",
        (
            "import os, sys; from pathlib import Path\n"
            "if os.path.lexists('STOPCAR'): sys.exit(3)\n"
            "Path('STOPCAR').write_text('LSTOP = .TRUE.\\n')"
        ),
    ]
    report = run_vasp(program, directory=tmp_path, dispatch=False)
    assert report.process.returncode == 0
    assert not (tmp_path / "STOPCAR").exists()
    assert report.soft_stop == {"mode": None, "requested_at": None, "lead_seconds": None, "acknowledged": None}


def test_invalid_soft_stop_or_stopcar_leaves_the_workdir_untouched(tmp_path: Path) -> None:
    outcar = tmp_path / "OUTCAR"
    outcar.write_text("previous run\n", encoding="utf-8")
    (tmp_path / "STOPCAR").write_text("LSTOP = .TRUE.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="soft_stop"):
        run_vasp(["true"], directory=tmp_path, dispatch=False, soft_stop="ionc")
    assert (tmp_path / "STOPCAR").is_file()
    (tmp_path / "STOPCAR").unlink()
    (tmp_path / "STOPCAR").symlink_to(outcar)
    with pytest.raises(ValueError, match="STOPCAR is not a regular file"):
        run_vasp(["true"], directory=tmp_path, dispatch=False)
    assert outcar.read_text(encoding="utf-8") == "previous run\n"
    assert (tmp_path / "STOPCAR").is_symlink()


def test_an_ionic_soft_stop_ends_a_relaxation_before_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = time.time()
    report = _stepping_mock(tmp_path, monkeypatch, soft_stop="ionic", stop_warning=now + 0.6, deadline=now + 60)
    assert report.classification == "stopped", report.diagnostics
    assert 1 <= _ionic_steps(tmp_path) < 20
    assert not (tmp_path / "STOPCAR").exists()
    record = report.soft_stop
    assert (record["mode"], record["acknowledged"]) == ("ionic", "soft")
    assert isinstance(record["requested_at"], float) and isinstance(record["lead_seconds"], float)
    assert 40 < record["lead_seconds"] < 60
    saved = json.loads((tmp_path / "vasp-run-report.json").read_text(encoding="utf-8"))
    assert saved["soft_stop"] == dict(record) and saved["classification"] == "stopped"
    mintime = soft_stop_mintime(report)
    assert mintime is not None and 80 <= _seconds(mintime) <= 120


def test_an_electronic_soft_stop_aborts_mid_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _stepping_mock(tmp_path, monkeypatch, soft_stop="electronic", stop_warning=time.time() + 0.6)
    assert report.classification == "stopped", report.diagnostics
    assert (report.soft_stop["mode"], report.soft_stop["acknowledged"]) == ("electronic", "hard")
    assert report.soft_stop["requested_at"] is not None and report.soft_stop["lead_seconds"] is None
    # No deadline, so no lead to derive a mintime from.
    assert soft_stop_mintime(report) is None


def test_no_known_time_writes_no_stopcar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _stepping_mock(tmp_path, monkeypatch, steps=3, pause=0.1, soft_stop="ionic")
    assert report.classification == "completed", report.diagnostics
    assert _ionic_steps(tmp_path) == 3
    assert report.soft_stop == {"mode": "ionic", "requested_at": None, "lead_seconds": None, "acknowledged": None}


def test_the_attempt_environment_supplies_the_times(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = int(time.time())
    monkeypatch.setenv("HTTK_WORKFLOW_STOP_WARNING", str(now - 1))
    monkeypatch.setenv("HTTK_WORKFLOW_DEADLINE", str(now + 100))
    report = _stepping_mock(tmp_path, monkeypatch, soft_stop="ionic")
    assert report.classification == "stopped", report.diagnostics
    lead = report.soft_stop["lead_seconds"]
    assert isinstance(lead, float) and 80 < lead <= 101
    # Without soft_stop the variables are not consulted at all.
    assert _stepping_mock(tmp_path, monkeypatch, steps=2, pause=0.1).soft_stop["mode"] is None


def test_an_acknowledged_stopcar_is_stopped_whoever_wrote_it(tmp_path: Path) -> None:
    outcar, oszicar = _relaxation(10, 3)
    report = run_vasp(
        _finishing(outcar, oszicar, stdout="Hard stop encountered!  aborting job ..."),
        directory=tmp_path,
        dispatch=False,
    )
    assert report.classification == "stopped"
    assert (report.soft_stop["mode"], report.soft_stop["acknowledged"]) == (None, "hard")
    assert soft_stop_mintime(report) is None
    # A soft stop before NSW without convergence is stopped, also without the footer.
    outcar, oszicar = _relaxation(10, 3, "soft stop encountered!  aborting job ...\n")
    report = run_vasp(_finishing(outcar.replace(_FOOTER, ""), oszicar), directory=tmp_path, dispatch=False)
    assert report.classification == "stopped", report.diagnostics


def test_the_evidence_reads_the_outcar_head_and_tails_only(tmp_path: Path) -> None:
    padding = "x" * 99 + "\n"
    outcar, oszicar = _relaxation(10, 3)
    head = outcar.removesuffix(_FOOTER)
    (tmp_path / "OSZICAR").write_text(oszicar, encoding="utf-8")
    # NSW is found past 2 MiB of header; a message more than 1 MiB before the end is not read.
    body = padding * 20_000 + head + padding * 20_000 + "Soft stop encountered!\n"
    (tmp_path / "OUTCAR").write_text(body + _FOOTER, encoding="utf-8")
    assert _stop_acknowledgement(tmp_path) == "soft" and _soft_stop_ended_early(tmp_path)
    (tmp_path / "OUTCAR").write_text(body + padding * 20_000 + _FOOTER, encoding="utf-8")
    assert _stop_acknowledgement(tmp_path) is None


@pytest.mark.parametrize(
    ("nsw", "steps", "extra", "expected"),
    [
        (0, 1, "", "completed"),
        (1, 1, "", "completed"),
        (10, 3, " reached required accuracy - stopping structural energy minimisation\n", "completed"),
        (3, 3, "", "nonconverged"),
    ],
)
def test_a_soft_stop_after_the_run_finished_keeps_its_classification(
    tmp_path: Path, nsw: int, steps: int, extra: str, expected: str
) -> None:
    outcar, oszicar = _relaxation(nsw, steps, extra, "soft stop encountered!  aborting job ...\n")
    report = run_vasp(_finishing(outcar, oszicar), directory=tmp_path, dispatch=False)
    assert report.classification == expected, report.diagnostics
    assert report.soft_stop["acknowledged"] == "soft"


@pytest.mark.parametrize(
    ("stdout", "options", "expected"),
    [
        ("hard stop encountered", {"returncode": 3}, "process_failure"),
        ("hard stop encountered\nZPOTRF failed", {"sleep": 0.5}, "diagnosed_stop"),
        ("hard stop encountered", {"sleep": 30}, "timeout"),
    ],
)
def test_stopped_never_overrides_a_failure(
    tmp_path: Path, stdout: str, options: dict[str, float], expected: str
) -> None:
    outcar, oszicar = _relaxation(10, 3)
    program = _finishing(outcar, oszicar, stdout=stdout, **options)  # type: ignore[arg-type]
    report = run_vasp(program, directory=tmp_path, dispatch=False, timeout=2, termination_grace=1)
    assert report.classification == expected
    assert report.soft_stop["acknowledged"] == "hard"


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def at(self, monitor: _SoftStop, now: float, event: str = "tick", line: str | None = None) -> None:
        self.now = now
        source = "process" if line is None else "OSZICAR"
        assert monitor(SourceEvent(event, source, line)) == ()  # type: ignore[arg-type]


_IONIC = "   1 F= -.10E+02 E0= -.10E+02  d E =-.1E-02"
_ELECTRONIC = "DAV:   1    -0.100000000000E+02   -0.10000E+02"


def test_the_monitor_writes_once_at_the_stop_warning(tmp_path: Path) -> None:
    clock = _Clock()
    monitor = _SoftStop("ionic", tmp_path, 1010.0, None, clock)
    clock.at(monitor, 1000.0, "start")
    clock.at(monitor, 1009.0)
    assert not (tmp_path / "STOPCAR").exists()
    clock.at(monitor, 1010.0)
    assert (tmp_path / "STOPCAR").read_text(encoding="utf-8") == "LSTOP = .TRUE.\n"
    assert monitor.requested_at == 1010.0
    (tmp_path / "STOPCAR").unlink()
    clock.at(monitor, 1020.0)
    assert not (tmp_path / "STOPCAR").exists() and monitor.requested_at == 1010.0


def test_the_monitor_writes_three_average_steps_before_the_deadline(tmp_path: Path) -> None:
    clock = _Clock()
    monitor = _SoftStop("electronic", tmp_path, 1090.0, 1100.0, clock)
    clock.at(monitor, 1000.0, "start")
    clock.at(monitor, 1010.0, "line", _ELECTRONIC)
    # An ionic line is no electronic step: the average stays 10 s, so the stop time 1070.
    clock.at(monitor, 1020.0, "line", _IONIC)
    assert monitor.steps == 1
    clock.at(monitor, 1069.0)
    assert monitor.requested_at is None
    clock.at(monitor, 1070.0)
    assert (tmp_path / "STOPCAR").read_text(encoding="utf-8") == "LABORT = .TRUE.\n"
    assert monitor.requested_at == 1070.0


def test_the_ionic_monitor_counts_only_ionic_oszicar_lines(tmp_path: Path) -> None:
    clock = _Clock()
    monitor = _SoftStop("ionic", tmp_path, None, 1100.0, clock)
    clock.at(monitor, 1000.0, "start")
    clock.at(monitor, 1001.0, "line", _ELECTRONIC)
    assert monitor(SourceEvent("line", "stdout", _IONIC)) == ()
    clock.at(monitor, 1099.0)
    # Neither a stop warning nor a completed step: nothing yet.
    assert monitor.steps == 0 and monitor.requested_at is None
    clock.at(monitor, 1099.0, "line", _IONIC)
    assert monitor.steps == 1 and monitor.requested_at == 1099.0


def test_the_monitor_never_writes_without_a_time(tmp_path: Path) -> None:
    clock = _Clock()
    monitor = _SoftStop("ionic", tmp_path, None, None, clock)
    clock.at(monitor, 1000.0, "start")
    for step in range(5):
        clock.at(monitor, 1001.0 + step, "line", _IONIC)
    clock.at(monitor, 1e12)
    assert monitor.requested_at is None and not (tmp_path / "STOPCAR").exists()


def test_an_unwritable_stopcar_leaves_the_run_supervised(tmp_path: Path) -> None:
    clock = _Clock()
    monitor = _SoftStop("ionic", tmp_path / "missing", 1000.0, None, clock)
    clock.at(monitor, 1000.0, "start")
    assert monitor.requested_at is None and monitor.done
