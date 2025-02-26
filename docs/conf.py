

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
import napari

sys.path.insert(0, os.path.abspath('..'))  # Ajoute le dossier racine de PythonTracer

project = 'PythonTracer'
copyright = '2025, Abdelghani Neuhaus'
author = 'Abdelghani Neuhaus'
release = '2025'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
		"sphinx.ext.autodoc",
		"sphinx.ext.autosummary",
		"sphinx.ext.autosectionlabel",
		"sphinx.ext.napoleon",
		"sphinx.ext.todo",
		"sphinx.ext.viewcode",
		"sphinx.ext.graphviz",
		"sphinxcontrib.jquery",
		]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
