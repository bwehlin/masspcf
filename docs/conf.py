# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'masspcf'
copyright = '2024-2025, Björn H. Wehlin'
author = 'Björn H. Wehlin'
release = '0.3.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'pydata_sphinx_theme'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# --- Create a temporary bundle of 'masspcf' and (a fake) 'mpcf_cpp' from the Python source in ../masspcf. This is only for documentation purposes (so that we don't have to keep reinstalling, including recompiling, masspcf everytime we want to update the docs). The setup has been tested on Linux and should probably work on OSX. It is unclear if it'll work on Windows.

import os
import sys

from glob import glob

import shutil

temp_mod_dir = os.path.abspath('modules')

os.makedirs(temp_mod_dir, exist_ok=True)

print(f'Will remove {temp_mod_dir}')

shutil.rmtree(temp_mod_dir, ignore_errors=True)
os.makedirs(os.path.join(temp_mod_dir, 'masspcf'))

masspcf_temp_dir = os.path.join(os.path.abspath('modules'), 'masspcf')


files = glob(os.path.abspath('../masspcf/*'))
for file in files: 
    target = os.path.join(masspcf_temp_dir, os.path.basename(file))
    os.symlink(file, target)

os.symlink(os.path.abspath('./mpcf_cpp'), os.path.join(masspcf_temp_dir, 'mpcf_cpp')) 

sys.path.insert(0, temp_mod_dir)
