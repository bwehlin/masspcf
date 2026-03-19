=========================================================================================================
masspcf: A computational package for discrete objects in Python and C++
=========================================================================================================

*masspcf* is a Python package with its backend written in C++ and CUDA for performing highly scalable computations involving piecewise constant functions (PCFs) and other discrete objects such as point clouds and persistence barcodes. The primary audience is practicioners within Topological Data Analysis (TDA) wanting to do statistical analysis on invariants such as *stable rank*, *Euler characteristic curves*, *Betti curves*, and so on.

The basic objects are *numpy*-like (multidimensional) arrays of PCFs, on which we support reductions such as taking averages across a dimension, etc. For 1-D arrays, we compute Lp distance matrices and L2 kernels that can then be used as input for, e.g., clustering, SVMs, and other machine learning algorithms.


.. toctree::
   :hidden:

   installing
   userguide
   acknowledgments

.. toctree::
   :hidden:
   :caption: Reference:

   masspcf

.. grid:: 2

   .. grid-item-card:: Getting started
      :link: installing
      :link-type: doc

      Install masspcf and get up and running quickly.

   .. grid-item-card:: User guide
      :link: userguide
      :link-type: doc

      In-depth guides on core concepts, tensors, distances, persistence, and GPU acceleration.

   .. grid-item-card:: API reference
      :link: masspcf
      :link-type: doc

      Detailed descriptions of the Python API.

   .. grid-item-card:: Citing masspcf
      :link: acknowledgments
      :link-type: doc

      How to cite masspcf in your research.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


