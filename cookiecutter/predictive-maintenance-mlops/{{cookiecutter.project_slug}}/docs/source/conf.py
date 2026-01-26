project = "Predictive Maintenance MLOps"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"

import os
import sys

# repo root so "import src.api.main" works
sys.path.insert(0, os.path.abspath("../.."))
