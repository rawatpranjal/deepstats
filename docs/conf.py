"""Sphinx configuration for deepstats documentation."""

import os
import sys

# Add package to path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "deepstats"
copyright = "2024, Pranjal Rawat"
author = "Pranjal Rawat"
version = "2.0.0"
release = "2.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

# MyST configuration for Markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "tasklist",
]
myst_heading_anchors = 3

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Napoleon settings (Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Intersphinx mappings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_title = "deepstats"

html_theme_options = {
    "github_url": "https://github.com/rawatpranjal/deepstats",
    "show_toc_level": 1,
    "navigation_with_keys": True,
    "show_prev_next": False,
}

# -- Source settings ---------------------------------------------------------
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

# -- Copy button settings ----------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
