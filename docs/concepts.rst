==============
Core Concepts
==============

This page introduces the foundational ideas behind masspcf: what piecewise constant functions are, how they are stored in tensors, and how the type system works.

Piecewise constant functions
============================

A **piecewise constant function** (PCF) is a function that takes a constant value on each of a finite number of intervals. For example, the function

.. math::

   f(t) = \begin{cases}
     1 & \text{if } 0 \le t < 2 \\
     3 & \text{if } 2 \le t < 5 \\
     0 & \text{if } 5 \le t < 7
   \end{cases}

is a PCF with three pieces.

In masspcf, a PCF is represented as an :math:`n \times 2` array of ``(time, value)`` pairs, where each row gives a breakpoint time and the value the function takes starting at that time. The function above would be represented as::

   [[0, 1],
    [2, 3],
    [5, 0]]

The value at each breakpoint is the value on the interval starting at that time and continuing until the next breakpoint (or until the end of the function's domain).

Why PCFs?
---------

Many invariants in **Topological Data Analysis** (TDA) are naturally piecewise constant. Examples include:

- **Stable rank functions**
- **Betti curves** -- the Betti number as a function of the filtration parameter
- **Euler characteristic curves** -- the Euler characteristic as a function of the filtration parameter

By representing these invariants as PCFs, masspcf enables efficient statistical analysis: computing means, distances, and norms over large collections of such functions, potentially leveraging GPU acceleration.

The Pcf class
--------------

An individual PCF is represented by :py:class:`~masspcf.Pcf`. You create one from a NumPy array or a list::

   import numpy as np
   import masspcf as mpcf

   # From a NumPy array
   data = np.array([[0.0, 1.0],
                     [2.0, 3.0],
                     [5.0, 0.0]], dtype=np.float32)
   f = mpcf.Pcf(data)

   # From a list (defaults to float32)
   g = mpcf.Pcf([[0, 1], [2, 3], [5, 0]])

You can convert a ``Pcf`` back to a NumPy array with :py:meth:`~masspcf.Pcf.to_numpy`::

   arr = f.to_numpy()  # shape (3, 2), dtype float32

Tensors
=======

While you *can* work with individual ``Pcf`` objects, masspcf is designed for working with **collections** of PCFs. These collections are stored in **tensors** -- multidimensional arrays, similar to NumPy's ``ndarray``.

A tensor can have any number of dimensions. For example:

- A 1-D tensor of shape ``(100,)`` holds 100 PCFs.
- A 2-D tensor of shape ``(10, 50)`` holds 500 PCFs arranged in a 10-by-50 grid.

Creating tensors
----------------

The primary way to create a tensor is with :py:func:`~masspcf.zeros`::

   import masspcf as mpcf

   # A 1-D tensor of 100 "zero" PCFs (32-bit, the default)
   X = mpcf.zeros((100,))

   # A 2-D tensor of 64-bit PCFs
   Y = mpcf.zeros((10, 50), dtype=mpcf.pcf64)

   # A tensor of scalar floats
   Z = mpcf.zeros((5, 5), dtype=mpcf.f32)

What "zero" means depends on the dtype: for PCF types, it is a function that is identically zero; for numeric types, it is the number 0; for point cloud types, it is an empty point cloud.

You can also generate random PCF tensors for experimentation::

   from masspcf.random import noisy_sin, noisy_cos

   # 200 noisy sin(2*pi*t) functions, each sampled at 100 time points
   sines = noisy_sin((200,), n_points=100)

   # A 2-D array: 10 x 50 noisy cosine functions
   cosines = noisy_cos((10, 50), n_points=30)

Indexing and slicing
---------------------

Tensors support NumPy-style indexing and slicing::

   X = mpcf.zeros((10, 5, 4))

   # Single element -- returns a Pcf
   f = X[3, 2, 1]

   # Slicing -- returns a tensor (view)
   row = X[3, :, :]        # shape (5, 4)
   sub = X[2:8, 1:, 2]     # shape (6, 4)

You can also assign into tensors::

   from masspcf.random import noisy_sin

   A = mpcf.zeros((2, 10))
   A[0, :] = noisy_sin((10,), n_points=100)
   A[1, :] = noisy_sin((10,), n_points=50)

Tensor types
------------

There are several concrete tensor types, each corresponding to a dtype:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Tensor class
     - dtype
     - Contents
   * - ``Pcf32Tensor``
     - ``pcf32``
     - 32-bit piecewise constant functions
   * - ``Pcf64Tensor``
     - ``pcf64``
     - 64-bit piecewise constant functions
   * - ``Float32Tensor``
     - ``f32``
     - 32-bit floating-point scalars
   * - ``Float64Tensor``
     - ``f64``
     - 64-bit floating-point scalars
   * - ``PointCloud32Tensor``
     - ``pcloud32``
     - 32-bit point clouds
   * - ``PointCloud64Tensor``
     - ``pcloud64``
     - 64-bit point clouds
   * - ``Barcode32Tensor``
     - ``barcode32``
     - 32-bit persistence barcodes
   * - ``Barcode64Tensor``
     - ``barcode64``
     - 64-bit persistence barcodes

In most cases, you do not need to construct these classes directly -- use :py:func:`~masspcf.zeros` or functions like :py:func:`~masspcf.random.noisy_sin` that return the appropriate tensor type automatically.


The dtype system
================

The ``dtype`` parameter controls the element type of a tensor, analogous to NumPy's ``dtype``. masspcf defines the following dtypes in :py:mod:`masspcf.typing` (also re-exported from the top-level ``masspcf`` module):

PCF types
---------

- :py:class:`~masspcf.pcf32` -- 32-bit piecewise constant functions (the default dtype)
- :py:class:`~masspcf.pcf64` -- 64-bit piecewise constant functions

Use ``pcf32`` for most work. Use ``pcf64`` when you need higher numerical precision.

Numeric types
-------------

- :py:data:`~masspcf.f32` -- 32-bit floating-point scalars (alias for ``numpy.float32``)
- :py:data:`~masspcf.f64` -- 64-bit floating-point scalars (alias for ``numpy.float64``)

These are used for tensors that hold scalar values, such as the results of norm or distance computations.

Point cloud types
-----------------

- :py:class:`~masspcf.pcloud32` -- 32-bit point clouds
- :py:class:`~masspcf.pcloud64` -- 64-bit point clouds

Used when working with point cloud data, e.g., as input to persistent homology computations.

Barcode types
-------------

- :py:class:`~masspcf.barcode32` -- 32-bit persistence barcodes
- :py:class:`~masspcf.barcode64` -- 64-bit persistence barcodes

Used to store persistence barcodes produced by homology computations.

Precision: 32-bit vs. 64-bit
------------------------------

Each dtype family comes in 32-bit and 64-bit variants. The 32-bit variants use less memory and are faster, especially on GPUs where single-precision throughput is typically much higher. Use 64-bit variants when numerical precision is important for your application.


CPU and GPU execution
=====================

masspcf automatically detects available NVIDIA GPUs and uses them for computations when beneficial. The library decides at runtime whether to execute a given operation on the CPU or GPU based on problem size.

You can query GPU availability::

   from masspcf import gpu

   gpu.has_nvidia_gpu()       # True/False
   gpu.nvidia_gpu_count()     # Number of GPUs
   gpu.detect_nvidia_gpus()   # Detailed GPU info

For more control, the :py:mod:`masspcf.system` module provides options to force CPU execution, limit the number of GPUs or CPU threads, and tune CUDA parameters. Most users will not need to change these settings.
