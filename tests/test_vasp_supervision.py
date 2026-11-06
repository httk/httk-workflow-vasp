import json
import sys
import time
from pathlib import Path

import pytest
from httk.workflow.codes import CheckerSpec, FollowSource

from httk.codes.vasp import VASP_CHECKS, VaspRunReport, parse_diagnostic_policy, read_checker_specs, run_vasp

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
