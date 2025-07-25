# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT

import inspect
import pkgutil
import pathlib
import packaging.version
import pyqg_jax

# Project information
project = "pyqg-jax"
copyright = "2023 Karl Otness"
author = "Karl Otness"
version = pyqg_jax.__version__
release = version

# Other configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx_rtd_theme",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]
html_css_files = ["css/pyqg-jax-fix-theme.css"]
suppress_warnings = ["epub.unknown_project_files"]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}

# Insert code into each rst file
rst_prolog = r"""

.. role:: pycode(code)
   :language: python

.. role:: cppcode(code)
   :language: cpp

"""

# Theme
html_theme = "sphinx_rtd_theme"

# MyST-NB configuration
nb_output_stderr = "remove"
nb_merge_streams = True
nb_execution_timeout = 180
nb_execution_raise_on_error = True
myst_enable_extensions = {"dollarmath"}
myst_dmath_double_inline = True

# Autodoc configuration
autodoc_mock_imports = []
autodoc_typehints = "none"
autodoc_member_order = "bysource"

# Napoleon configuration
napoleon_google_docstring = False

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyqg": ("https://pyqg.readthedocs.io/en/latest/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "powerpax": ("https://powerpax.readthedocs.io/en/latest/", None),
    "optax": ("https://optax.readthedocs.io/en/latest/", None),
    "equinox": ("https://docs.kidger.site/equinox/", None),
}


# Linkcode configuration
def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    mod_name = info["module"]
    if mod_name != "pyqg_jax" and not mod_name.startswith("pyqg_jax."):
        return None
    fullname = info["fullname"]
    pkg_root = pathlib.Path(pyqg_jax.__file__).parent
    try:
        obj = pkgutil.resolve_name(f"{mod_name}:{fullname}")
    except AttributeError:
        return None
    if isinstance(obj, property):
        obj = obj.fget
    if obj is None:
        return None
    obj = inspect.unwrap(obj)
    try:
        source_file = inspect.getsourcefile(obj)
        if source_file is None:
            return None
        source_file = pathlib.Path(source_file).relative_to(pkg_root)
        lines, line_start = inspect.getsourcelines(obj)
        line_end = line_start + len(lines) - 1
    except (ValueError, TypeError):
        return None
    # Form the URL from the pieces
    ref = (
        "master" if packaging.version.Version(version).is_devrelease else f"v{version}"
    )
    return (
        f"https://github.com/karlotness/pyqg-jax/blob/{ref}/src/pyqg_jax/"
        f"{source_file!s}#L{line_start}-L{line_end}"
    )
