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
- **`None` asks for VASP's own default.** An `incar_tags` value of `None` (JSON
  `null` in job parameters and in the Bash options file) removes the tag from the
  INCAR, the template's assignment included, and suppresses the value preparation
  would otherwise derive for it: `{"MAGMOM": null, "NBANDS": null}` leaves both to
  VASP. The band-count heuristic then reads VASP's default `MAGMOM` of 1 per ion
  for `ISPIN = 2`. A `null` `NBANDS` may still become explicit at launch: the
  launch wrapper writes VASP's default rounded to the band-group count when the
  launch has more than one band group and the rounded default stays within
  `2 × NELECT`, except in the runs it leaves to VASP and when
  `vasp.nbands_rounding` is `off` ([NBANDS rounding](#nbands-rounding)).
  Used on its own, {py:func}`~httk.codes.vasp.calculate_nbands` (Bash
  `httk_vasp_nbands`) reads the INCAR's `MAGMOM`, and without one falls back to
  the five-per-atom estimate of
  {py:func}`~httk.codes.vasp.suggested_magnetic_moments`, not VASP's 1 per ion;
  pass `magmom=` to say otherwise.
- **Updates rewrite statements, not lines.** {py:func}`~httk.codes.vasp.update_incar`
  understands the `;`-separated assignments {py:func}`~httk.codes.vasp.read_incar`
  reads, so updating one tag of `ISPIN = 2 ; ISYM = 2` leaves no second, stale
  assignment behind. A `None` value removes every assignment of its tag, for every
  caller, a remedy policy's `incar.<TAG>` change included; other statements on the
  line are kept and a line left empty disappears. In Bash,
  `httk_vasp_unset_tag TAG [INCAR]` (bridge command `vasp-unset-tag`) does the
  same and exits `0` without rewriting the file when the tag is absent;
  `httk_vasp_set_tag TAG null` writes the literal text `null`.
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
`bump_bands` for an INCAR without `NBANDS`, a KPOINTS change with no KPOINTS staged, or
with a KPOINTS that is not a Gamma or Monkhorst-Pack automatic mesh (line-mode and
explicit-list files are protected) — so
{py:func}`~httk.codes.vasp.apply_vasp_remedy` never receives a remedy it would
have to refuse. Both resolve a relative history path against that same directory;
{py:func}`~httk.codes.vasp.job_remedy_history_path` of the job payload is where a
runner should keep it, beside the job state, so an isolated workdir does not
silently restart the escalation. A staged runner forbids ladder steps that are
wrong for one stage (geometry changes, or `incar.ICHARG` in a non-self-consistent
stage) with `excluded_operations=`, which skips them like inapplicable steps.

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

## Launch wrapper

{py:mod}`httk.codes.vasp.launch` is the default way VASP starts. It runs once
per launch, in the attempt and in front of the launch prefix, where the
calculation's `INCAR`, `POSCAR`, `POTCAR` and `KPOINTS` and the attempt's rank
count are all known. It selects the VASP binary, plans the parallelization tags
`KPAR` and `NCORE`, writes them into the `INCAR`, rounds `NBANDS` to the
resulting band-group count, and then replaces itself with
the attempt's launch prefix (`HTTK_WORKFLOW_LAUNCH`) followed by the selected
binary:

```text
python -P -m httk.codes.vasp.launch --standard vasp_std --ncl vasp_ncl --gamma vasp_gam
```

{py:func}`~httk.codes.vasp.run_vasp`, and through it the Bash `httk_vasp_run`,
starts VASP this way by default, with its command as `--standard`, the
`vasp.ncl_command` and `vasp.gamma_command` settings as `--ncl` and `--gamma`
(its own command again as `--ncl` when that setting is unset, so a run keeps the
command it was given), and `--no-launch` for `launch=False`; `dispatch=False`
(`--no-dispatch`) bypasses the wrapper. Under atomate2 the VASP command mapping of
*httk-workflow*'s jobflow runner names the wrapper, in front of the launch
prefix, as all three `ATOMATE2_VASP_*CMD` commands, so atomate2 and the native
workflows plan their parallelization identically.

### Binary selection

{py:func}`~httk.codes.vasp.launch.select_vasp_command` reads `INCAR` and
`KPOINTS` in a directory and returns:

- the non-collinear command when `LNONCOLLINEAR` or `LSORBIT` is true; without
  one it raises `ValueError` naming the missing `--ncl`/`vasp.ncl_command`
  setting, rather than letting a standard binary fail on the INCAR;
- the Gamma-only command, when one is configured, for a run custodian's
  `_gamma_point_only_check` treats as Gamma-only: neither `LEPSILON` nor
  `LOPTICS` true, and `KPOINTS` an automatic Gamma or Monkhorst-Pack mesh of
  `1 1 1` whose shift is absent or zero. One deliberate deviation: with no
  `KPOINTS` file (a `KSPACING` run) the standard command is chosen, since the
  mesh is not judged from `KSPACING`. Explicit k-point lists, line mode and
  fully automatic files also select the standard command;
- the standard command otherwise.

Each command is one shlex-quoted string; `--ncl` and `--gamma` are optional.
`--directory` reads and updates the inputs in another directory (the command
still runs in the current one), and `--no-launch` execs the command without the
launch prefix.

### Parallelization planning

VASP splits its ranks into `KPAR` k-point groups and each group into
`NPAR = ranks / KPAR / NCORE` band groups, and rounds `NBANDS` up to a multiple
of `NPAR`. Left at VASP's defaults (`KPAR = NCORE = 1`) a small cell on many
ranks is rounded far above the bands it asked for: a diamond band structure
with `NBANDS = 39` on 32 ranks runs 64 bands, and custodian refuses a run whose
rounded `NBANDS` exceeds `2 × NELECT` as unrecoverable.
{py:func}`~httk.codes.vasp.parallel.plan_parallelization` decides, in this order:

1. **Explicit wins.** An `INCAR` that sets `NCORE`, `NPAR` or `KPAR` gets
   nothing written. When it also sets `NBANDS` and VASP's rounding under those
   tags would raise `NBANDS` above `2 × NELECT`, a warning says so and names
   custodian's refusal; another warns when the ranks do not split evenly into
   `KPAR` groups by `NPAR` (or `NCORE` when `NPAR` is unset). `NELECT` is the POTCAR `ZVAL` times the POSCAR
   counts, or the `INCAR`'s `NELECT`.
2. **Left to VASP.** DFPT and linear response (`LEPSILON`, `LCALCEPS`,
   `IBRION = 7, 8`), hybrid functionals (`LHFCALC`), `ALGO` values starting
   with `GW`, `BSE`, `ACFDT` or `RPA`, nudged elastic band runs (`IMAGES`), and
   single-rank runs get nothing.
3. **KPAR** is the largest divisor of the rank count that is at most a lower
   bound on the irreducible k-points and leaves every k-point group at least
   `ceil(NIONS / 2)` ranks, so a large cell keeps its ranks for bands and plane
   waves. The bound is the point count of an explicit list, segments × points of
   a line-mode file, and for a mesh (Gamma, Monkhorst-Pack, fully automatic, or
   derived from `KSPACING` (default 0.5, Gamma-centred unless `KGAMMA` is
   false) when there is no `KPOINTS`) the number of orbits of the lattice's
   point group that meet the mesh, the group found after Delaunay reduction of
   the POSCAR basis. That is a lower bound on VASP's irreducible count, since
   the crystal's point group is a subgroup of the lattice's; counting stops at
   the rank count. A mesh file with a nonzero shift uses the mesh size divided
   by 48, the largest point-group order. `LELF` and `LOPTICS` force `KPAR = 1`;
   `NCORE` is still planned for them.
4. **NCORE** is chosen among the divisors of the k-point group size that also
   divide the ranks per node when the binding says so. With an explicit
   `NBANDS` and NBANDS rounding on (the default), it is the largest candidate up
   to 8 whenever `NBANDS` rounded up to that `NPAR` stays at most
   `2 × NELECT`, and the wrapper then rounds `NBANDS` up by fewer than `NPAR`
   bands. Otherwise, and always with rounding `off`, it is the candidate closest
   to 8 (from below first) that keeps `NBANDS` a multiple of `NPAR`, or the
   whole group (`NPAR = 1`) when none does, so the planned tags never make VASP
   change an explicit `NBANDS`. Without one it is the largest candidate up to
   8, raised until VASP's default band count rounded up to `NPAR` is unchanged
   or stays at most `2 × NELECT`. That bound only keeps NPAR rounding from
   inflating the default: VASP prints "The number of bands has been changed"
   only for an `NBANDS` set in the INCAR, so custodian's refusal concerns the
   explicit case alone. The default is VASP's `main.F` formula,
   `max(floor((NELECT + 2) / 2) + max(floor(NIONS / 2), 3), floor(0.6 NELECT))`,
   plus `floor((ΣMAGMOM + 1) / 2)` for `ISPIN = 2` and doubled for a
   non-collinear run (pymatgen's `estimate_nbands` notes it differs from the
   VASP wiki).

A planned value of 1 is VASP's default and is not written, so the INCAR stays
minimal. In rounding mode an `NBANDS` the wrapper wrote counts as the value it
was rounded from (see below). The rank count is `--ranks` when given. Otherwise the launcher's own
flags win (the prefix's, or without a prefix the command's): `-n`, `-np` or
`--ntasks`, else `--ntasks-per-node` times the binding's node count. With a
prefix and no such flag, the attempt context's binding gives it (the sum of
each node's `procs`, whose common divisor bounds `NCORE`), else its
`resources` `procs`. Hybrid MPI/OpenMP runs (more than one thread per rank,
from an explicit `OMP_NUM_THREADS` in the prefix or the environment, else
`--cpus-per-task` or srun's `-c`), a thread flag without a rank count, a count
below 1, and a binding with GPUs leave it unknown. When it is unknown nothing is
planned, and the log line says why.

Planning is on unless `--parallel off` is passed or the setting
`vasp.parallelization` is `off`, resolved like other `vasp.*` settings: job
parameter, then the `HTTK_VASP_PARALLELIZATION` variable, then the workspace
setting, then `auto`. Any other value, including a boolean, is refused. Inputs the planner cannot
read leave the `INCAR` unchanged and VASP still starts.

### NBANDS rounding

After `KPAR` and `NCORE` are settled, planned or explicit, the wrapper rounds
`NBANDS` itself rather than leaving it to VASP, whose own adjustment interferes
with the `NCORE`/`KPAR` layout and, for an explicit `NBANDS`, trips custodian.
The band-group count is the `INCAR`'s explicit `NPAR`, else
`ranks / (KPAR × NCORE)` with the `INCAR`'s values (VASP's `1` for an absent
tag), so rounding also works with `vasp.parallelization` `off`. Then:

- an explicit `NBANDS` is rounded up to a multiple of `NPAR` when the result is
  at most `2 × NELECT`, and written only when that changes it; above the bound
  it is left alone with a warning (VASP then rounds it, and custodian refuses
  the run) that suggests removing `NBANDS` to restore VASP's default; this is
  the only warning about it, as the planner leaves it to the rounding step;
- an absent `NBANDS` becomes VASP's default (the `main.F` formula above, with
  its spin and non-collinear terms) rounded up to a multiple of `NPAR`, written
  only when it is at most `2 × NELECT`; otherwise it stays absent with a warning;
- nothing is rounded for `NPAR = 1` and for the runs the planner leaves to
  VASP (DFPT, `LHFCALC`, GW/BSE/ACFDT/RPA, `IMAGES`, one rank), which win over
  explicit tags; in these cases a value the wrapper wrote earlier still goes
  back to its base (an absent `NBANDS` for a `default` base), and the log says so;
- nothing at all is done when the rank count is unknown (hybrid MPI/OpenMP or
  GPU bindings included).

A value the wrapper writes carries a marker comment recording its base, for
example `NBANDS = 40  # httk: rounded from 37 for NPAR 8`, or
`NBANDS = 64  # httk: rounded from default 60 for NPAR 8` for VASP's default. On
the next launch a marked value that still equals its base rounded to the
recorded `NPAR` is recomputed from that base, so a run restarted on fewer ranks
shrinks back (to the base itself, or to an absent `NBANDS`, when the new count
no longer allows rounding) instead of ratcheting up; the same rank count rewrites
nothing. A value that differs from its marker, because a person or a remedy
edited it, is explicit and the marker is no longer trusted.
{py:func}`~httk.codes.vasp.update_incar`, and so every remedy, drops the marker
together with the value it replaces, and the wrapper removes marker lines left
on their own. A tool that rewrites the whole INCAR, such as custodian through
pymatgen's `Incar`, drops every comment, the marker included: from then on the
value counts as explicit, so the no-ratchet guarantee does not survive a
custodian correction, and a later launch on fewer ranks keeps (and at most
rounds) that value. Each decision is
logged as an `httk.codes.vasp.launch: nbands: ...` line naming the rank source,
the band-group count and the value before and after.

Rounding is on unless `--nbands-rounding off` is passed or the setting
`vasp.nbands_rounding` is `off`, resolved like `vasp.parallelization`: job
parameter, then `HTTK_VASP_NBANDS_ROUNDING`, then the workspace setting, then
`auto`; any other value is refused. With rounding `off` the planner keeps the
explicit-`NBANDS` rule that never changes it.

Every launch records its decision as one line
`httk.codes.vasp.launch: parallel: <reason>` on stderr (custodian keeps it in
the calculation directory's `std_err.txt`) and appends it, time-stamped, to
`httk-vasp-launch.log` in the directory, so the history of reruns stays
visible; the `nbands:` line follows, and warnings follow as `warning:` lines.
Because the written tags are explicit on the next launch, a rerun rewrites
nothing and logs that the INCAR sets them. The same rule makes custodian's own
fixes stick: `KPAR = 1` for `LELF`, `NCORE = 1` for `ZPOTRF` on small cells and
a doubled `NCORE` are all respected, a raised `NBANDS` is explicit (and only
rounded up to the band-group count within `2 × NELECT`; custodian's rewrite
drops the wrapper's marker, see above), and removing `NCORE`/`NPAR` for DFPT leaves a run
the planner does not touch. The flip side: a step that copies an earlier
step's INCAR (atomate2's `VASP_INHERIT_INCAR`, off by default) inherits that
step's planned tags, which may not suit its own rank count; leave inheriting
off, or set the tags explicitly for such runs. The INCAR keeps its file mode
and line endings when the tags are written.

When no command can be selected or executed, or `vasp.parallelization` or
`vasp.nbands_rounding` is invalid, the wrapper prints `httk.codes.vasp.launch: <reason>` to stderr and
exits with status 2; a program that cannot be executed is named together with
the variant (`--standard`, `--ncl` or `--gamma`) that selected it, or as the
launch prefix. Under atomate2 the job reports only "Job return code is 2":
custodian redirects stderr, so the reason is in `std_err.txt`. pymatgen is
never imported.

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
