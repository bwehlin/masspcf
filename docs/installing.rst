=====================
Installing masspcf
=====================

Installing using pip
=====================

The easiest way to get started with `masspcf` is using `pip`::

    pip install masspcf

.. note:: On OSX, ``masspcf`` is CPU-only.

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

To work with a collection of PCFs, store them in a 1-D ``Array``:

.. code-block:: python

    f1 = mpcf.Pcf(np.array([[0., 5.], [2., 3.], [5., 0.]]))
    f2 = mpcf.Pcf(np.array([[0., 2.], [4., 7.], [8., 1.], [9., 0.]]))
    f3 = mpcf.Pcf(np.array([[0.,  4.], [2., 3.], [3., 1.], [5., 0.]]))

    X = mpcf.Array([f1, f2, f3])

You can then compute pairwise :math:`L^p` distances or an :math:`L^2` kernel matrix in one call:

.. code-block:: python

    D = mpcf.pdist(X)       # pairwise L1 distance matrix (default p=1)
    K = mpcf.l2_kernel(X)   # L2 kernel matrix

Higher-dimensional arrays of PCFs are supported too, with NumPy-style indexing and reductions:

.. code-block:: python

    A = mpcf.zeros((4, 100))        # 4 x 100 array of zero PCFs
    mean = mpcf.mean(A, dim=1)      # mean along axis 1 → shape (4,)

See the :doc:`User guide <userguide>` for more detail on each of these topics.