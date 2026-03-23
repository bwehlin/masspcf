=========================================================================================================
masspcf: A computational package for discrete objects in Python and C++
=========================================================================================================

*masspcf* is a Python package with a C++/CUDA backend for GPU-accelerated computations on piecewise constant functions (PCFs) and related discrete objects such as point clouds, persistence barcodes, and distance matrices. The primary audience is practitioners within Topological Data Analysis (TDA) wanting to do statistical analysis on invariants such as *stable rank*, *Euler characteristic curves*, *Betti curves*, and so on.

The core data structures are NumPy-like multidimensional tensors supporting slicing, broadcasting, arithmetic, and reductions. Key operations include pairwise :math:`L_p` distance matrices, :math:`L_2` kernels, norms, and persistent homology (via Ripser). The resulting distance matrices and kernels can be used directly with scikit-learn and other machine learning libraries for clustering, classification, and dimensionality reduction.


.. toctree::
   :hidden:

   installing
   userguide

.. toctree::
   :hidden:
   :caption: Reference:

   masspcf
   changelog

.. toctree::
   :hidden:
   :caption: About:

   acknowledgments

.. grid:: 2

   .. grid-item-card:: :fas:`rocket` Getting started
      :link: installing
      :link-type: doc

      Install masspcf and get up and running quickly.

   .. grid-item-card:: :fas:`book` User guide
      :link: userguide
      :link-type: doc

      In-depth guides on core concepts, tensors, distances, persistence, and GPU acceleration.

   .. grid-item-card:: :fas:`code` API reference
      :link: masspcf
      :link-type: doc

      Detailed descriptions of the Python API.

   .. grid-item-card:: :fas:`circle-info` About
      :link: acknowledgments
      :link-type: doc

      Citing masspcf and acknowledgments.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


