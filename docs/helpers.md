# VASP helper API

*httk-workflow-vasp* ships the VASP helpers that native workflow runners are built
on, in two languages: the dependency-free Python package {py:mod}`httk.codes.vasp`
and the Bash VASP API, whose `httk_vasp_*` functions call the same code through the
*httk-workflow* shell bridge. The ready-made workflows built on them — `vasp.relax`,
`vasp.relax-bash`, `vasp.static` and `vasp.relax-static` — are published in
[workflows-vasp](https://github.com/httk/workflows-vasp); copy one of their runners
as the starting point of your own.

## Install

```console
python -m pip install httk-workflow-vasp
```

The distribution depends on *httk-core* and *httk-workflow*. Installing it
registers the `vasp` code through the `httk.registry.codes.vasp` registration
package, which is what makes the `vasp-*` bridge commands and the Bash API
available to every job the manager starts; nothing needs to be configured. The
result collectors additionally need the file readers of *httk-atomistic*.

## Python

A Python VASP runner is an ordinary {py:class}`~httk.workflow.Runner` whose steps
call the helpers directly:

```python
from httk.codes.vasp import (
    assemble_potcar,
    automatic_kpoint_grid,
    update_incar,
    write_automatic_kpoints,
)

grid = automatic_kpoint_grid(40, poscar="POSCAR")
write_automatic_kpoints(grid, "KPOINTS")
update_incar({"ENCUT": 520, "ISPIN": 2}, "INCAR")
assembled = assemble_potcar("/data/vasp/potpaw_PBE", poscar="POSCAR", output="POTCAR")
print([(item.species, item.variant, item.source) for item in assembled.choices])
```

The package covers:

- **Preparation** — {py:func}`~httk.codes.vasp.prepare_vasp_inputs` driven by
  {py:class}`~httk.codes.vasp.VaspPreparationOptions`, k-point grids
  ({py:func}`~httk.codes.vasp.automatic_kpoint_grid`,
  {py:func}`~httk.codes.vasp.write_automatic_kpoints`), POTCAR assembly
  ({py:func}`~httk.codes.vasp.assemble_potcar`), INCAR reading and updating,
  {py:func}`~httk.codes.vasp.calculate_nbands`, POSCAR normalization, scaling and
  rattling, and {py:func}`~httk.codes.vasp.contcar_to_poscar`.
- **Execution and diagnosis** — {py:func}`~httk.codes.vasp.run_vasp` supervises
  one VASP process and returns a classified
  {py:class}`~httk.codes.vasp.VaspRunReport`, removing a stale `OUTCAR` and
  `OSZICAR` first so an earlier run's errors cannot stop the new one;
  {py:func}`~httk.codes.vasp.diagnose_vasp_files` reads VASP 5 and 6 output into
  structured diagnostics without changing inputs;
  {py:func}`~httk.codes.vasp.validate_vasp_workdir` checks VASP's conservative
  240-byte workdir-path limit.
- **Extraction and cleanup** — the final energy, volume, POTIM and plane-wave
  count, and {py:func}`~httk.codes.vasp.clean_vasp_outputs` for explicit pre-run
  cleanup.

A few behaviors make a run reproducible and a retry meaningful ({doc}`details`
states them in full):

- **One k-point default.** {py:func}`~httk.codes.vasp.write_automatic_kpoints`,
  {py:class}`~httk.codes.vasp.VaspPreparationOptions` and the Bash API all start
  at `DEFAULT_KPOINT_CENTERING`, which is `Monkhorst-Pack`, because the remedy
  ladder promotes `Gamma` as the fix for two k-point failure classes.
- **Explicit tags win.** `VaspPreparationOptions.incar_tags` is written before
  anything is derived, and no derived value overwrites a tag the caller set.
- **A rattle needs entropy.** {py:func}`~httk.codes.vasp.rattle_poscar` requires
  an explicit `seed` or an `entropy` string, normally attempt-derived, because two
  retries that rattle identically are two identical calculations.
- **Cleanup keeps its own evidence.** {py:func}`~httk.codes.vasp.clean_vasp_outputs`
  preserves `CONTCAR` and `vasp-run-report.json` unless they are named in
  `also_remove`.
- **Provenance is recorded.** {py:func}`~httk.codes.vasp.assemble_potcar` returns a
  {py:class}`~httk.codes.vasp.PotcarAssembly` naming the variant, source, digest and
  `TITEL` of every chosen potential and writes the same record as
  `POTCAR.provenance.json`.

## Bash

A Bash runner sources the *httk-workflow* Bash API first and the VASP API second;
the manager exports both paths into every attempt:

```bash
source "$HTTK_WORKFLOW_BASH_API"
source "$HTTK_WORKFLOW_VASP_BASH_API"
httk_workflow_runner example.vasp prepare run

step_prepare() {
    httk_vasp_prepare_kpoints 40
    httk_vasp_set_tag ENCUT 520
    httk_workflow_advance run
}
```

The `httk_vasp_*` functions correspond directly to the Python helpers: `prepare`,
`prepare_kpoints`, `prepare_potcar`, `get_tag`, `set_tag` and `nbands`; `run` and `diagnose`;
`remedy_plan` and `remedy_apply`; `preclean`, `clean_outcar`, `normalize_poscar`,
`scale_poscar`, `rattle_poscar`, `promote_contcar`, and the energy, volume, POTIM,
plane-wave and POTCAR-summary extractors. Each is one `vasp-*` bridge command, and
every option of that command is available because arguments pass through
untouched. A legitimately absent answer, such as an INCAR tag that is not set,
exits `1`; a refused call exits `2` with the reason on stderr. `httk_vasp_run` and
`httk_vasp_diagnose` exit `20` on a structured diagnostic stop, `httk_vasp_run`
exits `21` when the calculation completed without converging, and
`httk_vasp_remedy_plan` exits `3` when the policy has no safe action left. A
complete Bash VASP runner is `vasp.relax-bash` of workflows-vasp.

## Remedy ladder

Remedies are never automatic: they are planned from diagnostics, then applied
explicitly.

```python
from httk.codes.vasp import apply_vasp_remedy, job_remedy_history_path, plan_vasp_remedy, run_vasp

report = run_vasp(["vasp_std"], directory=workdir)
history = job_remedy_history_path(payload)
decision = plan_vasp_remedy(report.diagnostics, directory=workdir, history_path=history)
if not decision.give_up:
    apply_vasp_remedy(decision, directory=workdir, history_path=history)
```

{py:func}`~httk.codes.vasp.plan_vasp_remedy` validates every rung of the ladder
against the calculation directory and skips one it cannot execute, so
{py:func}`~httk.codes.vasp.apply_vasp_remedy` never receives a remedy it would have
to refuse. Applying a decision uses a replayable workdir batch and records the
before and after input digests. The escalation history belongs beside the job
state ({py:func}`~httk.codes.vasp.job_remedy_history_path`), so a job with an
isolated workdir keeps climbing the ladder instead of silently restarting it.

Policies are a registry: {py:func}`~httk.codes.vasp.register_remedy_policy` adds
one, {py:func}`~httk.codes.vasp.remedy_policy_names` lists them, and an unknown
name is refused with the alternatives named. `reviewed-v1` is registered by the
package itself and is the default. A group whose practice differs registers its own
policy rather than editing the package. {doc}`details` documents the reviewed
ladder, including its ZHEGV recovery, in full.

## Collect

{py:mod}`httk.codes.vasp.collect` provides the result collectors of the
workflows-vasp packages: {py:func}`~httk.codes.vasp.collect.collect_vasp_relax`
(relaxed structure and total energy),
{py:func}`~httk.codes.vasp.collect.collect_vasp_static` (total energy) and
{py:func}`~httk.codes.vasp.collect.collect_vasp_relax_static` (relaxed structure
and the static stage's energy). Each takes one
{py:class}`~httk.workflow.collecting.JobRecord` and reads the job's committed data
when it has transactional data, or its persistent workdir otherwise; a
transactional job without committed data fails rather than falling back to
unpublished files. They need *httk-atomistic* for the VASP file readers.
