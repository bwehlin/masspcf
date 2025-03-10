=========================================================================================================
masspcf: A computational package for piecewise constant functions in Python and C++
=========================================================================================================

*masspcf* is a Python package with its backend written in C++ and CUDA for performing highly scalable computations involving piecewise constant functions (PCFs). The primary audience is practicioners within Topological Data Analysis (TDA) wanting to do statistical analysis on invariants such as *stable rank*, *Euler characteristic curves*, *Betti curves*, and so on.

The basic objects are *numpy*-like (multidimensional) arrays of PCFs, on which we support reductions such as taking averages across a dimension, etc. For 1-D arrays, we compute Lp distance matrices and L2 kernels that can then be used as input for, e.g., clustering, SVMs, and other machine learning algorithms.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installing
   acknowledgments
   tutorials

.. toctree::
   :maxdepth: 1
   :caption: Reference:

   masspcf

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


