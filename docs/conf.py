# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# Make the project root importable for autodoc (repo_root/)
DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(DOCS_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

# -- Project information -----------------------------------------------------
project = "DataPath"
author = "Tanviben Patel and Seyed Kahaki"
copyright = f"{datetime.now().year}, {author}"
release = "2.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Optional: todo settings -------------------------------------------------
todo_include_todos = True
