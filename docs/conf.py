# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import datetime
import tomllib;

pyproj_toml = tomllib.load(open('../pyproject.toml', 'rb'))

year = datetime.now().year

project = 'masspcf'
copyright = f'2024-{year}, Björn H. Wehlin'
author = 'Björn H. Wehlin'
release = pyproj_toml['project']['version']

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'sphinx.ext.imgmath',
              'sphinxcontrib.youtube',
              'breathe',
              'exhale']

breathe_projects = {
    "masspcf_internals": "./_build_doxygen/xml"
}

breathe_default_project = "masspcf_internals"

import textwrap

exhale_args = {
    # These arguments are required
    "containmentFolder":     "./_build_cpp_api",
    "rootFileName":          "library_root.rst",
    "doxygenStripFromPath":  "..",
    # Heavily encouraged optional argument (see docs)
    "rootFileTitle":         "masspcf C++ API reference",
    # Suggested optional arguments
    "createTreeView":        True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin":    "INPUT = ../include",
    "exhaleDoxygenStdin": textwrap.dedent('''
        INPUT      = ../include
    '''),


}


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
import platform
import shutil
import subprocess
from glob import glob

is_windows = platform.system() == "Windows"

temp_mod_dir = os.path.abspath('modules')

os.makedirs(temp_mod_dir, exist_ok=True)

print(f'Will remove {temp_mod_dir}')

shutil.rmtree(temp_mod_dir, ignore_errors=True)
os.makedirs(os.path.join(temp_mod_dir, 'masspcf'))

masspcf_temp_dir = os.path.join(os.path.abspath('modules'), 'masspcf')


files = glob(os.path.abspath('../masspcf/*'))
for file in files:
    if '__pycache__' in file:
        continue

    target = os.path.join(masspcf_temp_dir, os.path.basename(file))

    if is_windows:
        shutil.copy2(file, target)
    else:
        os.symlink(file, target)

# Handle the directory link/copy
cpp_src = os.path.abspath('./mpcf_cpp')
cpp_dest = os.path.join(masspcf_temp_dir, 'mpcf_cpp')

if is_windows:
    # Use copytree for directories; dirs_exist_ok=True prevents errors if it exists
    shutil.copytree(cpp_src, cpp_dest, dirs_exist_ok=True)
else:
    os.symlink(cpp_src, cpp_dest)

sys.path.insert(0, temp_mod_dir)

if False:
    # Doxygen for C++ docs

    try:
        retcode = subprocess.call("doxygen Doxyfile.in", shell=True)
        if retcode != 0:
            raise RuntimeError(f"Doxygen failed with retcode {retcode}")
    except OSError as e:
        raise OSError(f"Doxygen execution failed: {e}")
