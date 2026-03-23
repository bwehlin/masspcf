================
Getting started
================

Installing using pip
=====================

The easiest way to get started with `masspcf` is using `pip`::

    pip install masspcf

Git source build
=====================

In addition to the released versions of `masspcf` that are available on `PyPI`, it is also possible to install `masspcf` using `Git`. ::

    git clone https://github.com/bwehlin/masspcf.git
    cd masspcf

    # optional: select a specific tagged version to build
    git checkout tags/v0.3.1

    pip install .


On Windows and Linux, `masspcf` is built with CUDA enabled by default. To override this behavior, please set the environment variable ``BUILD_WITH_CUDA=0``. To force a CUDA build, set ``BUILD_WITH_CUDA=1``.

Quick start
===========

After installing, import ``masspcf`` and create your first piecewise constant function (PCF) from a NumPy array of ``(time, value)`` pairs:

.. code-block:: python

    import masspcf as mpcf
    import numpy as np

    # A PCF that equals 1 on [0,2), 3 on [2,5), and 0 on [5,7)
    f = mpcf.Pcf(np.array([[0, 1],
                            [2, 3],
                            [5, 0]]))

PCFs are callable -- you can evaluate them at any time:

.. code-block:: python

    f(1.0)    # 1.0  (on the interval [0, 2))
    f(3.5)    # 3.0  (on the interval [2, 5))

They also support arithmetic:

.. code-block:: python

    g = f * 2            # scale values by 2
    h = f + g            # pointwise addition
    s = f ** 0.5         # pointwise square root

Tensors: working with collections
----------------------------------

To work with a collection of PCFs, store them in a tensor created with ``mpcf.zeros``:

.. code-block:: python

    f1 = mpcf.Pcf(np.array([[0., 5.], [2., 3.], [5., 0.]]))
    f2 = mpcf.Pcf(np.array([[0., 2.], [4., 7.], [8., 1.], [9., 0.]]))
    f3 = mpcf.Pcf(np.array([[0., 4.], [2., 3.], [3., 1.], [5., 0.]]))

    X = mpcf.zeros((3,))
    X[0] = f1
    X[1] = f2
    X[2] = f3

For quick experimentation, generate random data:

.. code-block:: python

    from masspcf.random import noisy_sin, noisy_cos

    sines   = noisy_sin((200,), n_points=100)   # 200 noisy sin functions
    cosines = noisy_cos((10, 50), n_points=30)  # 10 x 50 noisy cosines

Distances, norms, and reductions
---------------------------------

Compute pairwise :math:`L^p` distances and norms:

.. code-block:: python

    D = mpcf.pdist(X)               # pairwise L1 distance matrix
    norms = mpcf.lp_norm(X, p=1)    # L1 norm of each PCF

Higher-dimensional tensors support NumPy-style reductions:

.. code-block:: python

    A = mpcf.zeros((4, 100))
    avg = mpcf.mean(A, dim=1)       # mean along axis 1 → shape (4,)

Persistent homology
--------------------

Compute persistent homology from point cloud data and convert the resulting barcodes to PCF summaries:

.. code-block:: python

    from masspcf.persistence import (compute_persistent_homology,
                                      barcode_to_stable_rank,
                                      barcode_to_betti_curve)

    # Point cloud: 50 random points in R^3
    pts = mpcf.zeros((1,), dtype=mpcf.pcloud32)
    pts[0] = np.random.rand(50, 3).astype(np.float32)

    barcodes = compute_persistent_homology(pts)    # Ripser
    sr = barcode_to_stable_rank(barcodes)          # stable rank as a PCF
    bc = barcode_to_betti_curve(barcodes)          # Betti curve as a PCF

Saving and loading
-------------------

Save tensors to disk and load them back:

.. code-block:: python

    from masspcf.io import save, load

    save(X, 'my_pcfs.mpcf')
    X_loaded = load('my_pcfs.mpcf')

GPU acceleration
-----------------

masspcf automatically uses NVIDIA GPUs when available. You can check GPU status and control execution:

.. code-block:: python

    from masspcf import gpu

    gpu.has_nvidia_gpu()             # True/False
    gpu.nvidia_gpu_count()           # number of available GPUs

See the :doc:`User guide <userguide>` for more detail on each of these topics.

.. note:: On OSX, ``masspcf`` is CPU-only.
