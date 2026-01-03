project = "Predictive Maintenance MLOps"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
templates_path = ["_templates"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"

import os
import sys
sys.path.insert(0, os.path.abspath("../.."))
