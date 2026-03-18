=======================
Persistent Homology
=======================

This guide covers the persistent homology pipeline in masspcf: going from point cloud data to persistence barcodes to stable rank functions, and using those functions for downstream analysis.


Background
==========

**Persistent homology** is a tool from Topological Data Analysis (TDA) that captures the "shape" of data at multiple scales. Given a set of points (a point cloud), persistent homology tracks the appearance and disappearance of topological features -- connected components, loops, voids, etc. -- as a scale parameter increases.

The output is a **persistence barcode**: a collection of intervals :math:`[b_i, d_i)`, where each interval records the birth :math:`b_i` and death :math:`d_i` of a topological feature. Features that persist over a wide range of scales are considered significant, while short-lived features are often regarded as noise.

masspcf provides two functional summaries of persistence barcodes, both of which are piecewise constant functions:

- The (1d) **stable rank** counts, for each threshold :math:`t`, how many bars have length at least :math:`t` [CR20]_ [GC17]_ [SCL17]_.
- The **Betti curve** counts, for each filtration value :math:`t`, how many bars are alive at :math:`t` (see, e.g., [U17]_ [CM21]_).

Because these summaries are PCFs, they fit naturally into masspcf's tensor framework, enabling efficient computation of distances and means over collections of barcodes.


The pipeline
============

A typical TDA workflow in masspcf follows three steps:

1. **Point clouds** -- Organize your data into a tensor of point clouds
2. **Barcodes** -- Compute persistent homology to get persistence barcodes
3. **Functional summaries** -- Convert barcodes to stable rank or Betti curve PCFs for further analysis

.. code-block:: text

   Point clouds  ──>  Barcodes  ──>  Stable ranks / Betti curves (PCFs)
                                          │
                                          ├──  distances (pdist)
                                          ├──  means (mean)
                                          └──  norms (lp_norm)


Step 1: Point clouds
====================

Point clouds are stored in ``PointCloud32Tensor`` or ``PointCloud64Tensor``. Each element of the tensor is a point cloud, represented internally as an :math:`n \times d` numeric array where :math:`n` is the number of points and :math:`d` is the ambient dimension.

Create a tensor and assign point clouds as NumPy arrays::

   import masspcf as mpcf
   import numpy as np

   # A tensor that will hold 5 point clouds
   pclouds = mpcf.zeros((5,), dtype=mpcf.pcloud64)

   # Assign random point clouds (varying number of points, 3-dimensional)
   for i in range(5):
       n_points = np.random.randint(20, 100)
       pclouds[i] = np.random.randn(n_points, 3)

Point clouds in the same tensor can have different numbers of points and, in principle, different dimensions, though in practice it is most common for all point clouds to share the same ambient dimension.

Higher-dimensional tensors work as well::

   # A 10 x 20 grid of point clouds
   pclouds = mpcf.zeros((10, 20), dtype=mpcf.pcloud64)


Step 2: Computing persistent homology
=======================================

:py:func:`~masspcf.persistence.compute_persistent_homology` takes a tensor of point clouds and returns a tensor of persistence barcodes. Barcode computation is performed using Ripser [B21]_ under the hood::

   from masspcf import persistence as mpers

   bcs = mpers.compute_persistent_homology(pclouds, maxDim=1)

The ``maxDim`` parameter controls the highest homology dimension computed. With ``maxDim=1``, the function computes :math:`H_0` (connected components) and :math:`H_1` (loops).

The output tensor has one extra dimension appended, of size ``maxDim + 1``. For example:

- Input shape ``(5,)`` with ``maxDim=1`` produces output shape ``(5, 2)``
- Input shape ``(10, 20)`` with ``maxDim=2`` produces output shape ``(10, 20, 3)``

To access the :math:`H_n` barcode for a specific point cloud, index with the point cloud's position followed by ``n``::

   bc_H0 = bcs[3, 0]   # H0 barcode of point cloud 3
   bc_H1 = bcs[3, 1]   # H1 barcode of point cloud 3

Each element is a :py:class:`~masspcf.persistence.Barcode` object.

Input flexibility
-----------------

``compute_persistent_homology`` also accepts:

- A single ``Float32Tensor`` or ``Float64Tensor`` (interpreted as a single point cloud)
- A plain NumPy array (interpreted as a single point cloud)

::

   # From a NumPy array directly
   points = np.random.randn(50, 3)
   bcs = mpers.compute_persistent_homology(points, maxDim=1)

Options
-------

The function supports the following options:

- ``distance_type`` -- The distance metric used between points. Currently only ``DistanceType.Euclidean`` (the default).
- ``complex_type`` -- The simplicial complex construction. Currently only ``ComplexType.VietorisRips`` (the default).
- ``reduced`` -- If ``True``, compute reduced homology. If ``False`` (the default), an essential ``[0, inf)`` bar is added to :math:`H_0` representing the single connected component that never dies. This matches the convention used by most TDA textbooks.
- ``verbose`` -- Print progress information (default ``True``).


Step 3: Functional summaries
=============================

Persistence barcodes can be converted to piecewise constant functions for
downstream analysis. Because the results are PCFs, they fit naturally into
masspcf's tensor framework, enabling distances, means, and norms.

Stable ranks
-------------

:py:func:`~masspcf.persistence.barcode_to_stable_rank` converts barcodes into
stable rank PCFs. The stable rank counts, for each threshold :math:`t`, the
number of bars with length (death minus birth) strictly greater than :math:`t`
[CR20]_::

   sranks = mpers.barcode_to_stable_rank(bcs)

The output tensor has the same shape as the input.

Betti curves
-------------

:py:func:`~masspcf.persistence.barcode_to_betti_curve` converts barcodes into
Betti curves. The Betti curve counts, for each filtration value :math:`t`, the
number of bars alive at :math:`t` (i.e., bars with birth :math:`\leq t <`
death)::

   bettis = mpers.barcode_to_betti_curve(bcs)

The output tensor has the same shape as the input.

Using functional summaries
---------------------------

Since stable ranks and Betti curves are PCFs, they are stored in PCF tensors
and support all of masspcf's standard operations::

   import masspcf as mpcf
   from masspcf.plotting import plot as plotpcf
   import matplotlib.pyplot as plt

   # Plot the H1 stable ranks
   plotpcf(sranks[:, 1])
   plt.title('H1 stable ranks')
   plt.show()

   # Compute distances between H1 stable ranks
   D = mpcf.pdist(sranks[:, 1], verbose=False)

   # Compute the mean H1 stable rank
   avg = mpcf.mean(sranks[:, 1], dim=0)


Complete example
================

The following example creates a multidimensional tensor of random point clouds, computes persistent homology, converts to stable ranks, and visualizes the result::

   import masspcf as mpcf
   from masspcf import persistence as mpers
   from masspcf.plotting import plot as plotpcf
   import numpy as np
   import matplotlib.pyplot as plt

   shape = (10, 20)
   pcloud_dim = 4

   # Create and fill point cloud tensor
   pclouds = mpcf.zeros(shape, dtype=mpcf.pcloud64)
   for i in range(shape[0]):
       for j in range(shape[1]):
           n_points = np.random.randint(20, 100)
           pclouds[i, j] = np.random.randn(n_points, pcloud_dim)

   # Compute persistent homology (H0 and H1)
   bcs = mpers.compute_persistent_homology(pclouds, maxDim=1)
   print(bcs.shape)   # (10, 20, 2)

   # Convert to stable ranks
   sranks = mpers.barcode_to_stable_rank(bcs)
   print(sranks.shape) # (10, 20, 2)

   # Plot H1 stable ranks for the first row of point clouds
   plotpcf(sranks[0, :, 1])
   plt.title('H1 stable ranks for pclouds[0, :, :]')
   plt.show()

   # Distance matrix between H1 stable ranks in the first row
   D = mpcf.pdist(sranks[0, :, 1], verbose=False)


The Barcode class
=================

Individual barcodes are represented by :py:class:`~masspcf.persistence.Barcode`. A barcode can be constructed from an :math:`n \times 2` NumPy array of ``(birth, death)`` pairs::

   from masspcf.persistence import Barcode

   bc = Barcode(np.array([[0.0, 1.5],
                           [0.2, 3.0],
                           [0.5, 0.8]]))

You can also convert a single barcode to a stable rank::

   from masspcf.persistence import barcode_to_stable_rank

   sr = barcode_to_stable_rank(bc)
   # sr is a Pcf


References
==========

.. [B21] Bauer, U. (2021). Ripser: efficient computation of Vietoris–Rips persistence barcodes. *Journal of Applied and Computational Topology*, 5(3), 391–423.

.. [CR20] Chachólski, W., & Riihimäki, H. (2020). Metrics and stabilization in one parameter persistence. *SIAM Journal on Applied Algebra and Geometry*, 4(1), 69–98.

.. [GC17] Gäfvert, O., & Chachólski, W. (2017). Stable invariants for multiparameter persistence. *arXiv preprint* arXiv:1703.03632.

.. [SCL17] Scolamiero, M., Chachólski, W., Lundman, A., Ramanujam, R., & Öberg, S. (2017). Multidimensional persistence and noise. *Foundations of Computational Mathematics*, 17, 1367–1406.

.. [U17] Umeda, Y. (2017). Time series classification via topological data analysis. *Information and Media Technologies*, 12, 228–239.

.. [CM21] Chazal, F., & Michel, B. (2021). An introduction to topological data analysis: fundamental and practical aspects for data scientists. *Frontiers in Artificial Intelligence*, 4, 667963.
