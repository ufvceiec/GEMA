import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "GEMA"
author = "Alberto Nogales, Álvaro José García-Tejedor"
release = "0.4.3"
version = "0.4.3"
copyright = f"2024, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# GitHub Pages: prevent Jekyll from interfering with Sphinx output
html_extra_path = []  # populated below at build time


def setup(app):
    import pathlib
    out = pathlib.Path(app.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".nojekyll").touch()
