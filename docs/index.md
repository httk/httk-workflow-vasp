# *httk-workflow-vasp*

This site documents the *httk-workflow-vasp* module. For the full documentation
of *httk₂*, see [docs.httk.org](https://docs.httk.org).

The module adds VASP support to *httk-workflow*: the dependency-free Python
helpers in `httk.codes.vasp` (input preparation, supervised execution,
structured diagnosis, the reviewed remedy ladder, and result collectors), the
Bash VASP API a Bash runner sources as `$HTTK_WORKFLOW_VASP_BASH_API`, and the
`vasp-*` bridge commands behind that API. Installing it registers the `vasp`
code with *httk₂* through the `httk.registry.codes.vasp` registration package.

The ready-made VASP workflows built on these helpers live in the
[workflows-vasp](https://github.com/httk/workflows-vasp) repository.

```{admonition} Quick links
:class: tip

- {doc}`helpers` — the VASP helper API in Python and Bash
- {doc}`details` — reproducibility behaviors and the full ZHEGV recovery ladder
- {doc}`reference/index` — the generated API reference
```

## Install

```console
python -m pip install httk-workflow-vasp
```

```{toctree}
:maxdepth: 2
:caption: Documentation

helpers
details
reference/index
```
