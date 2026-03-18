======================
Distances and Norms
======================

masspcf provides GPU-accelerated computation of pairwise distance matrices and norms for collections of piecewise constant functions. These are key building blocks for downstream machine learning tasks such as clustering, classification, and dimensionality reduction.


Mathematical background
========================

Given two PCFs :math:`f` and :math:`g`, the :math:`L_p` distance between them is

.. math::

   d_p(f, g) = \left( \int_0^\infty |f(t) - g(t)|^p \, dt \right)^{1/p}.

The :math:`L_p` norm of a single PCF :math:`f` is

.. math::

   \| f \|_p = \left( \int_0^\infty |f(t)|^p \, dt \right)^{1/p}.

For piecewise constant functions, these integrals reduce to finite sums over the breakpoint intervals, which masspcf evaluates exactly.


Pairwise distances
==================

:py:func:`~masspcf.pdist` computes the full pairwise :math:`L_p` distance matrix for a 1-D tensor of PCFs. The result is a symmetric :math:`n \times n` NumPy array where entry :math:`(i, j)` is the :math:`L_p` distance between the :math:`i`-th and :math:`j`-th PCF.

Basic usage
-----------

::

   import masspcf as mpcf
   from masspcf.random import noisy_sin

   X = noisy_sin((50,), n_points=100)

   # L1 distance matrix (default)
   D = mpcf.pdist(X)
   # D.shape is (50, 50)

The ``p`` parameter controls which :math:`L_p` distance is computed::

   D1 = mpcf.pdist(X, p=1)   # L1 distance
   D2 = mpcf.pdist(X, p=2)   # L2 distance

Progress output
---------------

By default, ``pdist`` prints progress information during computation. To suppress this, pass ``verbose=False``::

   D = mpcf.pdist(X, verbose=False)

Input requirements
------------------

``pdist`` requires a **1-D** PCF tensor. If your data is in a higher-dimensional tensor, slice out the dimension you want first::

   A = mpcf.zeros((2, 50))
   # ... fill A ...

   # Distances between the 50 PCFs in the first row
   D = mpcf.pdist(A[0, :])

GPU acceleration
-----------------

For large collections of PCFs, ``pdist`` automatically offloads the computation to the GPU when one is available. The number of pairwise integrals grows as :math:`n(n-1)/2`, so GPU acceleration can provide dramatic speedups for large :math:`n`.


Symmetric matrices
===================

:py:class:`~masspcf.SymmetricMatrix` provides a compressed storage format for symmetric matrices. Internally it stores only the lower triangle — :math:`n(n+1)/2` elements instead of :math:`n^2` — while supporting transparent ``[i, j]`` access.

::

   from masspcf import SymmetricMatrix
   from masspcf.typing import f32

   m = SymmetricMatrix(100, dtype=f32)
   m[3, 7] = 2.5
   assert m[7, 3] == 2.5   # symmetric access

To convert to a full NumPy array::

   dense = m.to_dense()   # shape (100, 100), dtype float32


Norms
=====

:py:func:`~masspcf.lp_norm` computes the :math:`L_p` norm of every PCF in a tensor, returning a NumPy array of the same shape.

::

   import masspcf as mpcf
   from masspcf.random import noisy_sin

   X = noisy_sin((50,), n_points=100)

   # L1 norms of all 50 PCFs
   norms = mpcf.lp_norm(X, p=1)
   # norms.shape is (50,)

For higher-dimensional tensors, the output shape matches the input::

   A = mpcf.zeros((3, 10))
   # ... fill A ...

   norms = mpcf.lp_norm(A, p=1)
   # norms.shape is (3, 10)


Using distances in machine learning
=====================================

The distance matrices produced by ``pdist`` are standard NumPy arrays, making them easy to use with scikit-learn and other libraries.

Clustering
----------

::

   from sklearn.cluster import AgglomerativeClustering

   D = mpcf.pdist(X, verbose=False)

   clustering = AgglomerativeClustering(
       n_clusters=3,
       metric='precomputed',
       linkage='average'
   )
   labels = clustering.fit_predict(D)

Multidimensional scaling
-------------------------

::

   from sklearn.manifold import MDS

   D = mpcf.pdist(X, verbose=False)

   mds = MDS(n_components=2, dissimilarity='precomputed')
   coords = mds.fit_transform(D)

Classification with distance-based kernels
--------------------------------------------

A common approach is to convert a distance matrix into a kernel matrix and use it with a kernel SVM. For example, using a Gaussian (RBF) kernel:

.. math::

   K_{ij} = \exp\!\left(-\frac{D_{ij}^2}{2\sigma^2}\right)

::

   import numpy as np
   from sklearn.svm import SVC

   D = mpcf.pdist(X, verbose=False)
   sigma = np.median(D[D > 0])
   K = np.exp(-D**2 / (2 * sigma**2))

   clf = SVC(kernel='precomputed')
   clf.fit(K, labels)
