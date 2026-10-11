"""The VASP half of the Bash authoring SDK.

A managed Bash runner finds the VASP API at the path the manager exports and
calls a VASP verb inside a real attempt, and the ``vasp-*`` bridge commands,
mounted through the ``codes`` registry tier, keep the bridge's absent and refused
exit codes.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from httk.workflow import TaskManager, Workspace
from httk.workflow.collecting import job_records
from httk.workflow.scaffold import new_job

_VASP_RUNNER = """#!/usr/bin/env bash
set -euo pipefail
: "${HTTK_WORKFLOW_VASP_BASH_API:?}"
source "$HTTK_WORKFLOW_BASH_API"
source "$HTTK_WORKFLOW_VASP_BASH_API"
httk_workflow_runner tests.bash.vasp start

step_start() {
    printf 'ENCUT = 520\\n' >INCAR
    httk_workflow_state_set encut "$(httk_vasp_get_tag ENCUT)"
    httk_workflow_succeed
}

httk_workflow_main
"""


def test_a_managed_bash_runner_sources_the_vasp_api_and_calls_a_vasp_verb(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path / "workspace")
    package = tmp_path / "package"
    package.mkdir()
    (package / "httk_workflow.toml").write_text(
        '[workflow]\nname = "tests.bash.vasp"\n\n[workflow.runner]\nsteps = ["start"]\n', encoding="utf-8"
    )
    (package / "run").write_text(_VASP_RUNNER, encoding="utf-8")
    (package / "run").chmod(0o755)
    job = new_job(workspace, package, placement="bash/jobs", install=True)
    with TaskManager(workspace, heartbeat_interval=0.01) as manager:
        manager.run_until_idle(timeout=120.0)

    [record] = job_records(workspace, states=("succeeded", "failed"))
    assert (record.job_id, record.state) == (job.job_id, "succeeded"), record.failure
    root = record.workspace_root / record.payload_path
    assert json.loads((root / ".httk-job" / "state.json").read_text(encoding="utf-8")) == {"encut": 520}


def _bridge(cwd: Path, *arguments: str, stdin: str | None = None) -> "subprocess.CompletedProcess[str]":
    environment = {name: value for name, value in os.environ.items() if not name.startswith("HTTK_WORKFLOW_")}
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    return subprocess.run(
        [sys.executable, "-m", "httk.workflow._shell_bridge", *arguments],
        cwd=cwd,
        env=environment,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )


def test_code_bridge_commands_keep_the_absent_and_refused_exit_codes(tmp_path: Path) -> None:
    (tmp_path / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
    (tmp_path / "POSCAR").write_text("Si\n1.0\n1 0 0\n0 1 0\n0 0 1\nSi\n1\nDirect\n0 0 0\n", encoding="utf-8")
    refusal = "httk-workflow: vasp-rattle-poscar needs --seed or --entropy"

    absent = _bridge(tmp_path, "vasp-get-tag", "NSW")
    assert (absent.returncode, absent.stdout, absent.stderr) == (1, "", "")
    refused = _bridge(tmp_path, "vasp-rattle-poscar")
    assert refused.returncode == 2 and refused.stderr.startswith(refusal), refused.stderr

    batch_absent = _bridge(tmp_path, "batch", stdin="vasp-get-tag ENCUT\nvasp-get-tag NSW\n")
    assert (batch_absent.returncode, batch_absent.stdout) == (1, "520\n")
    assert batch_absent.stderr == "httk-workflow: batch line 2 failed: vasp-get-tag NSW\n"
    batch_refused = _bridge(tmp_path, "batch", stdin="vasp-rattle-poscar\n")
    assert batch_refused.returncode == 2
    assert batch_refused.stderr.startswith(refusal) and "batch line 1 failed" in batch_refused.stderr


def test_the_reapply_bridge_command_prints_the_replayed_changes(tmp_path: Path) -> None:
    (tmp_path / "INCAR").write_text("ISYM = 2\n", encoding="utf-8")
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps({"events": [{"changes": [{"operation": "incar.ISYM", "value": 0}]}]}), encoding="utf-8"
    )

    result = _bridge(tmp_path, "vasp-remedy-reapply", "--history", str(history))

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [{"operation": "incar.ISYM", "value": 0}]
    assert (tmp_path / "INCAR").read_text(encoding="utf-8") == "ISYM = 0\n"


def test_the_report_problems_bridge_command_lists_distinct_non_warning_codes(tmp_path: Path) -> None:
    diagnostics = [
        {"code": "ionic_nonconvergence", "severity": "error"},
        {"code": "lreal_false", "severity": "warning"},
        {"code": "zpotrf", "severity": "fatal"},
        {"code": "ionic_nonconvergence", "severity": "error"},
    ]
    (tmp_path / "vasp-run-report.json").write_text(json.dumps({"diagnostics": diagnostics}), encoding="utf-8")

    result = _bridge(tmp_path, "vasp-report-problems", "vasp-run-report.json")

    assert (result.returncode, result.stdout) == (0, "ionic_nonconvergence\nzpotrf\n"), result.stderr
    assert _bridge(tmp_path, "vasp-report-problems", "missing.json").returncode == 2
