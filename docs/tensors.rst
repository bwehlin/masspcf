======================
Working with Tensors
======================

This guide covers the practical details of creating, manipulating, and persisting tensors in masspcf.

Creating tensors
================

Using zeros
-----------

The most common way to create a tensor is :py:func:`~masspcf.zeros`, which allocates a tensor of a given shape filled with "zero" elements::

   import masspcf as mpcf

   # 1-D tensor of 100 PCFs (32-bit, the default)
   X = mpcf.zeros((100,))

   # 3-D tensor of 64-bit PCFs
   Y = mpcf.zeros((4, 10, 25), dtype=mpcf.pcf64)

   # Scalar float tensor
   Z = mpcf.zeros((5, 5), dtype=mpcf.float64)

For PCF dtypes, "zero" is a function that is identically zero. For numeric dtypes, it is the number 0. For point cloud dtypes, it is an empty point cloud.

Generating random data
-----------------------

For quick experimentation, :py:mod:`masspcf.random` provides functions that generate tensors of noisy trigonometric PCFs::

   from masspcf.random import noisy_sin, noisy_cos

   # 200 noisy sin(2*pi*t) functions, each with 100 breakpoints
   sines = noisy_sin((200,), n_points=100)

   # 2-D: 10 x 50 noisy cosine functions with 30 breakpoints each
   cosines = noisy_cos((10, 50), n_points=30)

These functions return ``PcfTensor`` by default. Pass ``dtype=mpcf.pcf64`` for 64-bit.

From serialized NumPy data
---------------------------

:py:func:`~masspcf.from_serial_content` constructs a tensor from PCF data already stored in NumPy arrays — a flat content array and an enumeration array that describes how to split it::

   import numpy as np
   import masspcf as mpcf

   # Three PCFs packed into a single content array
   content = np.array([
       [0.0, 2.5], [1.5, 1.2], [3.14, 0.0],   # PCF 0 (3 points)
       [0.0, 7.0], [3.8, 5.5], [4.5, 1.5], [7.0, 0.0],  # PCF 1 (4 points)
       [0.0, 3.0], [2.0, 0.0],                   # PCF 2 (2 points)
   ])

   # Each row gives (start, end) indices into content
   enumeration = np.array([[0, 3], [3, 7], [7, 9]])

   F = mpcf.from_serial_content(content, enumeration)
   # F is a PcfTensor of shape (3,)

The enumeration array can be multidimensional. If it has shape ``(n1, n2, ..., nk, 2)``, the resulting tensor has shape ``(n1, n2, ..., nk)``.


Shape and copying
=================

Every tensor has a :py:attr:`shape` property, along with ``ndim``, ``size``,
and ``len()`` — matching the NumPy interface::

   X = mpcf.zeros((10, 5, 4))
   X.shape        # (10, 5, 4)
   X.ndim         # 3
   X.size         # 200
   len(X)         # 10  (first axis)

To create an independent copy (not a view)::

   Y = X.copy()

To collapse all dimensions into one::

   flat = X.flatten()  # shape (200,)

To change the shape without changing the data, use ``reshape``. One dimension
may be ``-1`` to infer its size::

   X = mpcf.FloatTensor(np.arange(12, dtype=np.float32))
   X.reshape((3, 4))     # shape (3, 4)
   X.reshape((2, -1))    # shape (2, 6) — inferred

For contiguous tensors, ``reshape`` returns a view (shared data). For
non-contiguous tensors (e.g. from slicing with a step), it copies first.

To reverse the order of axes, use the ``.T`` property. For finer control,
``transpose`` accepts an explicit axis permutation::

   A = mpcf.FloatTensor(np.arange(12, dtype=np.float32).reshape(3, 4))
   A.T              # shape (4, 3)
   A.transpose((1, 0))  # same as .T for 2-D

   B = mpcf.FloatTensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
   B.transpose((2, 0, 1))  # shape (4, 2, 3)

Transpose always returns a view.

To swap exactly two axes, use ``swapaxes``::

   C = mpcf.FloatTensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
   C.swapaxes(0, 2)     # shape (4, 3, 2)
   C.swapaxes(-1, -3)   # same — negative indices count from the last axis

To remove size-1 dimensions, use ``squeeze``. With no argument it removes all
of them; with an axis argument it removes only that one::

   X = mpcf.FloatTensor(np.arange(6, dtype=np.float32).reshape(1, 6, 1))
   X.squeeze()      # shape (6,)
   X.squeeze(0)     # shape (6, 1)

Squeeze always returns a view. Squeezing an axis whose size is not 1 raises
``ValueError``.

The inverse operation, ``expand_dims``, inserts a size-1 dimension at the given
position (negative indexing supported)::

   Y = mpcf.FloatTensor(np.arange(6, dtype=np.float32))
   Y.expand_dims(0)    # shape (1, 6)
   Y.expand_dims(-1)   # shape (6, 1)

Expand dims also returns a view.


Type casting
============

``astype`` converts a tensor to a different dtype. Same-family precision changes
(e.g. float32 to float64) and numeric cross-family casts (e.g. int to float)
are supported::

   X = mpcf.FloatTensor(np.array([1.5, 2.5, 3.5], dtype=np.float32))
   X.astype(mpcf.float64)    # FloatTensor, float64
   X.astype(mpcf.int32)      # IntTensor, int32 (truncates)

PCF and point cloud tensors support precision changes within their family::

   F = mpcf.zeros((5,), dtype=mpcf.pcf32)
   F.astype(mpcf.pcf64)      # PcfTensor, pcf64

``astype`` always returns a new tensor (copy).


Joining tensors
===============

``concatenate`` joins tensors along an existing axis::

   A = mpcf.FloatTensor(np.array([[1, 2], [3, 4]], dtype=np.float32))  # (2, 2)
   B = mpcf.FloatTensor(np.array([[5, 6]], dtype=np.float32))          # (1, 2)
   mpcf.concatenate((A, B), axis=0)   # (3, 2)

All tensors must have the same shape except along the join axis.

``stack`` joins tensors along a new axis (all shapes must match)::

   X = mpcf.FloatTensor(np.array([1, 2, 3], dtype=np.float32))  # (3,)
   Y = mpcf.FloatTensor(np.array([4, 5, 6], dtype=np.float32))  # (3,)
   mpcf.stack((X, Y), axis=0)    # (2, 3)
   mpcf.stack((X, Y), axis=1)    # (3, 2)


Splitting tensors
=================

``split`` divides a tensor into parts along an axis. Pass an integer for equal
splits, or a list of indices for custom split points::

   X = mpcf.FloatTensor(np.arange(12, dtype=np.float32).reshape(4, 3))

   # Equal split: 4 rows into 2 parts of 2 rows each
   a, b = mpcf.split(X, 2, axis=0)       # each shape (2, 3)

   # Index split: split at rows 1 and 3
   p, q, r = mpcf.split(X, [1, 3], axis=0)  # shapes (1,3), (2,3), (1,3)

The returned parts are views sharing data with the original tensor. An equal
split raises ``ValueError`` if the axis size is not divisible by the number of
sections.

``array_split`` works the same way but allows uneven divisions — the first
sections get one extra element when the size is not evenly divisible::

   Y = mpcf.FloatTensor(np.arange(9, dtype=np.float32))
   parts = mpcf.array_split(Y, 4)   # sizes: 3, 2, 2, 2


Iterating
=========

Iterating over a tensor yields sub-tensors along the first axis, just like
NumPy::

   X = mpcf.FloatTensor(np.arange(12, dtype=np.float32).reshape(3, 4))
   for row in X:
       print(row.shape)   # (4,)

For a 1-D tensor, iteration yields scalar elements.

This also enables ``list()``, ``tuple()``, and unpacking::

   a, b, c = X   # three rows

Nested iteration works as expected::

   Y = mpcf.FloatTensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
   for matrix in Y:          # shape (3, 4)
       for row in matrix:    # shape (4,)
           print(row)
