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
from httk.workflow.protocol import JobSpec, prepare_job_payload

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
    payload = tmp_path / "payload"
    runner = payload / "files" / "runner"
    runner.parent.mkdir(parents=True)
    runner.write_text(_VASP_RUNNER, encoding="utf-8")
    runner.chmod(0o755)
    job = prepare_job_payload(
        payload,
        JobSpec(name="VASP bash runner", workflow="tests.bash.vasp", runner_path="files/runner", initial_step="start"),
    )
    workspace.submit(payload, "bash/jobs")
    with TaskManager(workspace, heartbeat_interval=0.01) as manager:
        manager.run_until_idle(timeout=120.0)

    marker = workspace.find_marker_by_id(job.id)
    assert marker is not None and marker.kind == "succeeded"
    root = workspace.payload_path(marker.placement, marker.job_key)
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
