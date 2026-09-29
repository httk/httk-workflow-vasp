# VASP helper details

This page states the behaviors of {py:mod}`httk.codes.vasp` that make a run
reproducible and a retry meaningful, and documents the reviewed ZHEGV recovery
ladder in full. {doc}`helpers` is the practical overview.

## Preparation and cleanup

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

{py:func}`~httk.codes.vasp.validate_vasp_workdir` checks VASP's conservative path
limit and {py:func}`~httk.codes.vasp.clean_vasp_outputs` performs explicit pre-run
cleanup while preserving requested files. Diagnosis never changes inputs.

- **One k-point default.** {py:func}`~httk.codes.vasp.write_automatic_kpoints`,
  {py:class}`~httk.codes.vasp.VaspPreparationOptions`, and the Bash bridge all
  start at `DEFAULT_KPOINT_CENTERING`, which is `Monkhorst-Pack`. The reviewed
  remedy ladder promotes `Gamma` as the fix for two k-point failure classes, so a
  workflow that already started there would have no such remedy left.
- **Explicit tags win.** `VaspPreparationOptions.incar_tags` is written to the INCAR
  before anything is derived, and no derived value overwrites a tag the caller set.
- **Updates rewrite statements, not lines.** {py:func}`~httk.codes.vasp.update_incar`
  understands the `;`-separated assignments {py:func}`~httk.codes.vasp.read_incar`
  reads, so updating one tag of `ISPIN = 2 ; ISYM = 2` leaves no second, stale
  assignment behind.
- **A rattle needs entropy.** {py:func}`~httk.codes.vasp.rattle_poscar` requires an
  explicit `seed` or an `entropy` string that {py:func}`~httk.codes.vasp.derive_seed`
  turns into one — normally something attempt-derived — because two retries that
  rattle identically are two identical calculations.
- **Cleanup keeps its own evidence.** {py:func}`~httk.codes.vasp.clean_vasp_outputs`
  preserves `VASP_RESTART_ARTIFACTS`, that is `CONTCAR` and `vasp-run-report.json`,
  which are what the remedy machinery and restart promotion read. Name them in
  `also_remove` to delete them anyway.
- **Provenance is recorded.** {py:func}`~httk.codes.vasp.assemble_potcar` returns a
  {py:class}`~httk.codes.vasp.PotcarAssembly` naming the variant, source path,
  digest, and `TITEL` of every chosen potential, and writes the same record next to
  the POTCAR as `POTCAR.provenance.json`.

## Remedies

Remedies are bounded, planned, and applied explicitly.
{py:func}`~httk.codes.vasp.plan_vasp_remedy` validates every rung of the ladder
against the calculation directory it is given and skips one it cannot execute — a
`bump_bands` for an INCAR without `NBANDS`, a KPOINTS change with no KPOINTS staged
— so {py:func}`~httk.codes.vasp.apply_vasp_remedy` never receives a remedy it would
have to refuse. Both resolve a relative history path against that same directory;
{py:func}`~httk.codes.vasp.job_remedy_history_path` of the job payload is where a
runner should keep it, beside the job state, so an isolated workdir does not
silently restart the escalation.

Policies are a registry rather than module source:
{py:func}`~httk.codes.vasp.register_remedy_policy` adds one,
{py:func}`~httk.codes.vasp.remedy_policy_names` lists what is registered, and an
unknown name is refused with the alternatives named. `reviewed-v1` is registered by
the package itself and is the default.

### ZHEGV recovery

`reviewed-v1` includes `edddav_zhegv`, retaining the historical diagnostic code
for both `EDDAV: Call to ZHEGV failed` and `EDDDAV: Call to ZHEGV failed`.
The case-insensitive pattern is
`\b(?:Error\s+)?EDD?DAV\s*:\s*Call\s+to\s+ZHEGV\s+failed\b`;
it accepts optional `Error`, variable whitespace and boxed output without
depending on a particular return code. Actual examples are the boxed,
unprefixed [VASP 6.6 EDDAV message](https://www.vasp.at/forum/viewtopic.php?t=20669)
and the prefixed [VASP 5.4.1 EDDDAV message](https://www.vasp.at/forum/viewtopic.php?p=21798).
The motivating report was a one-atom carbon cell (mp-998866) on 32 MPI ranks,
with `Returncode = 42 2 64`, which reportedly ran on about four ranks.
That observation motivates a decomposition remedy; neither the string nor its
return code proves the cause.

The bounded CPU MPI ladder is:

1. `incar.NPAR = 1`: eliminate parallel band groups, distributing each orbital's
   plane-wave coefficients across the available ranks in its k-point group.
   One divides any group size, so the planner needs no scheduler topology.
   [NPAR takes precedence over NCORE](https://www.vasp.at/wiki/NPAR), including
   inherited NCORE settings. This directly targets excessive band decomposition
   before adding orbitals. It can be very slow: this is a recovery setting,
   not a recommended production layout, and does not reduce the allocation.
2. `bump_bands = 2`: add two orbitals to an explicit INCAR `NBANDS`, retaining
   `NPAR=1`. Extra empty states can help iterative eigensolver convergence
   ([NBANDS](https://www.vasp.at/wiki/NBANDS)). This reuses the `too_few_bands`
   operation but keeps separate diagnosis and history: ZHEGV has other causes.
   The increment is from INCAR, not the previous run's MPI-rounded effective
   count. If NBANDS is absent, the planner skips this step and gives up; it
   does not guess from electron count or stale output. Standard runner preparation
   normally supplies NBANDS from the staged structure and POTCAR.

Both operations already exist in {py:func}`~httk.codes.vasp.apply_vasp_remedy`; no
new operation is needed. The ZHEGV precedence is below geometry/factorization and
memory diagnostics and above generic electronic nonconvergence and
`too_few_bands`. Each application records the next step in job-scoped history
before changing inputs. At most two ZHEGV retries occur; persistent failure gives
`give_up` at step 2, even if other lower-priority diagnostics coexist. The
runner's `maximum_remedies` can stop earlier. Even an already-present `NPAR=1`
consumes its rung once, never loops.

No ALGO change is included: `Normal` already uses blocked Davidson, while
`All` may also start with Davidson and is incompatible with potential-only
functionals ([ALGO](https://www.vasp.at/wiki/ALGO)). The ladder does not change
the geometry, Hamiltonian settings, convergence thresholds or completion checks.
The original error and each edit remain recorded; a retry is successful only
if the usual runner completion checks pass. This is not a proof of the root cause
or of physical convergence. It does not repair bad geometries, corrupt restart
files, an ill-conditioned overlap from another cause, compiler/BLAS/ScaLAPACK
defects, memory exhaustion, or GPU/OpenMP placement. Plane-wave decomposition
can itself remain excessive. GPU/threaded builds may ignore or constrain these
parallelization tags; this ladder is not a reviewed remedy for those layouts.

Stub tests cover classification, precedence, actual input edits, job-history
escalation, manager recovery and terminal failure. They do not validate VASP
numerics. The end-to-end check is to rerun mp-998866 with VASP 6.6 on 32 ranks,
without the manual rank reduction, using this library revision, and inspect
the applied ladder, effective parallelization and converged output. That
cluster check remains to be performed by the supervisor/operator.

## Dependencies

These functions are independent *httk₂* interfaces. They do not import the Python
*httk* v1 `httk.task.ht_tasks_api` or `httk.task.vasptools` implementations. The
VASP modules were audited to use only the Python standard library, *httk-core*,
and *httk-workflow*: its code-support toolkit `httk.workflow.codes` and, for the
collectors, `httk.workflow.collecting.JobRecord`. No numeric or
atomistic capability package is required; only the result collectors need
*httk-atomistic*. Legacy Unix-compress `POTCAR.Z` is the one external-format
exception: it is read through a checked argv invocation of `gzip` or `uncompress`
when either executable is available.
