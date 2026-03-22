==================
Saving and loading
==================

masspcf provides a binary format for efficiently saving and loading tensors. All tensor types are supported, including PCF, numeric, point cloud, barcode, symmetric matrix tensors, etc.

Saving
======

Use :py:func:`~masspcf.save` to write a tensor to a file::

   import masspcf as mpcf
   from masspcf.random import noisy_sin

   X = noisy_sin((100,), n_points=50)
   mpcf.save(X, 'my_pcfs.mpcf')

You can also pass an open file object in binary write mode::

   with open('my_pcfs.mpcf', 'wb') as f:
       mpcf.save(X, f)

Pickle support
==============

All tensor types and standalone objects (``Pcf``, ``Barcode``, ``DistanceMatrix``,
``SymmetricMatrix``) support Python's ``pickle`` protocol. This means they work
with ``pickle.dumps``/``pickle.loads``, ``copy.deepcopy``, and multiprocessing::

   import pickle

   data = pickle.dumps(X)
   X_restored = pickle.loads(data)

Pickling uses masspcf's binary format internally, so it is efficient and
preserves dtype and shape.

.. note::

   Many masspcf operations (distance matrices, reductions, etc.) are already
   parallelized internally using multithreading and GPU acceleration. Layering
   Python ``multiprocessing`` on top will most likely *decrease* performance in
   these cases due to process overhead and memory duplication.


Loading
=======

Use :py:func:`~masspcf.load` to read a tensor back::

   X = mpcf.load('my_pcfs.mpcf')

The returned tensor will be of the same type and dtype as what was saved. As with ``save``, you can also pass an open file object::

   with open('my_pcfs.mpcf', 'rb') as f:
       X = mpcf.load(f)
