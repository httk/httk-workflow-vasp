# httk-workflow-vasp

![Status: Early beta](https://img.shields.io/badge/status-early--beta-orange)

> **⚠️ EARLY BETA**
>
> This is an early beta release of *httk₂*. The organization of the packages
> and their APIs should not yet be regarded as stable, and may change between
> releases.

*httk-workflow-vasp* adds VASP support to
[*httk-workflow*](https://github.com/httk/httk-workflow), the workflow engine of
[*httk₂*](https://github.com/httk/httk2). It provides `httk.codes.vasp`, a
dependency-free Python library for VASP input preparation, supervised execution,
structured diagnosis, a reviewed remedy ladder and result collection, and the
Bash VASP API that exposes the same helpers to Bash runners. Installing it
registers the `vasp` code with *httk₂*; nothing needs to be configured.

## Install

```console
python -m pip install httk-workflow-vasp
```

## Use

In a Python runner:

```python
from httk.codes.vasp import automatic_kpoint_grid, update_incar, write_automatic_kpoints

write_automatic_kpoints(automatic_kpoint_grid(40, poscar="POSCAR"), "KPOINTS")
update_incar({"ENCUT": 520, "ISPIN": 2}, "INCAR")
```

In a Bash runner, whose manager exports the path of the VASP API:

```bash
source "$HTTK_WORKFLOW_BASH_API"
source "$HTTK_WORKFLOW_VASP_BASH_API"
httk_vasp_prepare_kpoints 40
encut=$(httk_vasp_get_tag ENCUT)
```

Ready-made VASP workflows built on these helpers are published in
[workflows-vasp](https://github.com/httk/workflows-vasp). The helper API is
documented in [`docs/helpers.md`](docs/helpers.md) and at
[docs.httk.org/httk-workflow-vasp](https://docs.httk.org/httk-workflow-vasp/).

## Running tests

`make test` runs the normal profile; `make ci` runs formatting, lint, both type
checkers and the extended tests. The collector tests skip unless
*httk-atomistic* is installed.
