import importlib
import os
import warnings
from datetime import date

from sphinx.deprecation import RemovedInSphinx10Warning

warnings.filterwarnings("ignore", category=RemovedInSphinx10Warning)

project = "httk-workflow-vasp"
author = "The httk-workflow-vasp AUTHORS"
copyright = f"{date.today().year}, {author}"

extensions = [
    # Core API docs
    "sphinx.ext.autodoc",        # pull docstrings
    "sphinx.ext.autosummary",    # API summary tables + stub gen
    "sphinx.ext.napoleon",       # Google/NumPy docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",        # math rendering via MathJax

    # Nice-to-haves
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",

    # Markdown + notebooks
    "myst_nb",                   # .ipynb support

    "autoapi.extension",
    "httk.core.docs.sphinx_ext",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**/.ipynb_checkpoints"]

# Autosummary: generate stub pages automatically
autosummary_generate = True

# Autodoc defaults (tweak to taste)
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"  # no-op under AutoAPI 3.8 (annotations render fully qualified); kept for intent
typehints_fully_qualified = False
typehints_document_rtype = True
typehints_defaults = "comma"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_attr_annotations = True

# MyST / Markdown configuration (math + nice syntax)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
    "dollarmath",  # enables $...$ and $$...$$
]
myst_heading_anchors = 3

# Execute examples during strict builds; docs-clean also clears the cache.
nb_execution_mode = "cache"
nb_execution_raise_on_error = True

html_theme = "furo"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# External references resolve against inventories vendored in docs/_inventories/
# so docs builds need no network access; link targets still point at the live
# sites. Refresh the committed inventories with `make docs-inventories`.
#
# When this module cross-references public objects from another httk
# distribution (e.g. httk.core), add it here against the published httk docs
# site. The base URL comes from the DOCS_BASE_URL Makefile variable (exported as
# HTTK_DOCS_BASE_URL); the default below keeps bare sphinx invocations working.
# Vendor each dependency inventory alongside python.inv, for example:
#     "httk-core": (f"{_docs_base_url}/httk-core/", "_inventories/httk-core.inv"),
_docs_base_url = os.environ.get("HTTK_DOCS_BASE_URL", "https://docs.httk.org")

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", "_inventories/python.inv"),
    "httk-core": (f"{_docs_base_url}/httk-core/", "_inventories/httk-core.inv"),
    "httk-workflow": (f"{_docs_base_url}/httk-workflow/", "_inventories/httk-workflow.inv"),
}

autoapi_options = [
       "members",
       "undoc-members",
       "show-inheritance",
       "show-module-summary",
       "imported-members",
]
autoapi_root = "reference/autoapi"
autoapi_ignore = []  # scan everything; the reference is curated by skip_member below

autoapi_type = "python"
autoapi_dirs = ["../src/httk"]
autoapi_add_toctree_entry = False
autoapi_keep_files = True
autoapi_member_order = "bysource"
autoapi_python_class_content = "module"  # docstring under class, not merged from __init__
autoapi_python_use_implicit_namespaces = True
autoapi_template_dir = "_templates/autoapi"

nitpicky = True
nitpick_ignore = [
    ("py:class", "typing.Any"),
    ("py:class", "typing.Optional"),
    ("py:class", "typing.Union"),
    ("py:class", "Ellipsis"),
]

# The public facade :mod:`httk.codes.vasp` re-exports the surface of cohesive
# internal modules that get no reference page, but AutoAPI resolves annotations
# against the defining module, so those cross-references cannot land and are
# ignored here. The private type aliases the same signatures mention are ignored
# for the same reason.
_INTERNAL_MODULES = ("inputs", "diagnostics", "remedies", "reports")
nitpick_ignore_regex = [
    (r"py:.*", r"httk\.codes\.vasp\.(" + "|".join(_INTERNAL_MODULES) + r")(\..+)?"),
    (r"py:.*", r"(DiagnosticSeverity|RemedyChange|RemedySequence)"),
]
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

suppress_warnings = ["myst.xref_missing", "autoapi.python_import_resolution"]

# The API reference documents a deliberate public surface, not every source
# object. Only the modules named here get a reference page; everything else (the
# internal modules behind the facade, the bridge, the registration package) is
# still scanned so the facade can document the names it re-exports. Within a
# documented module, only the names it lists in ``__all__`` appear, which drops
# the toolkit types the package merely imports.
PUBLIC_MODULES = frozenset({"httk.codes.vasp", "httk.codes.vasp.collect"})

_exports_cache: dict[str, frozenset[str] | None] = {}


def _module_exports(module_name):
    """Return the ``__all__`` of one module, or ``None`` when it declares none."""

    if module_name not in _exports_cache:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # pragma: no cover - a module that will not import
            _exports_cache[module_name] = None
        else:
            names = getattr(module, "__all__", None)
            _exports_cache[module_name] = None if names is None else frozenset(names)
    return _exports_cache[module_name]


def skip_member(app, what, name, obj, skip, options):
    obj_id = str(getattr(obj, "id", None) or name)
    if what in {"module", "package"}:
        # Only the deliberate public modules get a page; the rest stay scanned
        # (so re-exports resolve) but unrendered.
        return obj_id not in PUBLIC_MODULES
    if name.startswith("_"):
        return True
    owner, _, short = obj_id.rpartition(".")
    exports = _module_exports(owner)
    if exports is not None and short not in exports:
        return True
    return skip


def setup(sphinx):
    sphinx.connect('autoapi-skip-member', skip_member)
