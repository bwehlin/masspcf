==================
Saving and loading
==================

masspcf provides a binary format for efficiently saving and loading tensors. All tensor types are supported, including PCF, numeric, point cloud, and barcode tensors.

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

Loading
=======

Use :py:func:`~masspcf.load` to read a tensor back::

   X = mpcf.load('my_pcfs.mpcf')

The returned tensor will be of the same type and dtype as what was saved. As with ``save``, you can also pass an open file object::

   with open('my_pcfs.mpcf', 'rb') as f:
       X = mpcf.load(f)
