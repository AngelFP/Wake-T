# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import date
sys.path.insert(0, os.path.abspath('../..'))


# -- Import Wake-T version ---------------------------------------------------
from wake_t import __version__  # noqa: E402


# -- Project information -----------------------------------------------------
project = 'Wake-T'
project_copyright = '2019-%s, Ángel Ferran Pousa' % date.today().year
author = 'Ángel Ferran Pousa'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx_design',
    'sphinx_gallery.gen_gallery',
    'numpydoc'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'  # "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Logo
html_logo = "_static/logo.png"
html_favicon = "_static/favicon_128x128.png"

# Theme options
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/AngelFP/Wake-T",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Slack",
            "url": "https://wake-t.slack.com/",
            "icon": "fa-brands fa-slack",
        },
    ],
    "use_edit_page_button": True,    
    "pygment_light_style": "default",
    "pygment_dark_style": "monokai",
}

html_context = {
    "github_user": "AngelFP",
    "github_repo": "Wake-T",
    "github_version": "dev",
    "doc_path": "docs/source",
}

# Do not show type hints.
autodoc_typehints = 'none'

# Do  not use numpydoc to generate autosummary.
numpydoc_show_class_members = False

# Create autosummary for all files.
autosummary_generate = True

# Autosummary configuration
autosummary_context = {
    # Methods that should be skipped when generating the docs
    "skipmethods": ["__init__"]
}

# Configuration for generating tutorials.
from sphinx_gallery.sorting import FileNameSortKey  # noqa: E402

sphinx_gallery_conf = {
     'examples_dirs': '../../tutorials',
     'gallery_dirs': 'tutorials',
     'filename_pattern': '.',
     'within_subsection_order': FileNameSortKey,
}

# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3/", None),
#     "numpy": ("https://numpy.org/devdocs/", None),
# }
