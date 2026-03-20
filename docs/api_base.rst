masspcf
=======

Core library for piecewise constant functions, tensors, and computations.

pcf
---

.. automodule:: masspcf.pcf
   :members:
   :undoc-members:
   :show-inheritance:

tensor
------

.. automodule:: masspcf.tensor
   :members:
   :undoc-members:
   :show-inheritance:

tensor_create
-------------

.. automodule:: masspcf.tensor_create
   :members:
   :undoc-members:
   :show-inheritance:

reductions
----------

.. automodule:: masspcf.reductions
   :members:
   :undoc-members:
   :show-inheritance:

distance
--------

.. automodule:: masspcf.distance
   :members:
   :undoc-members:
   :show-inheritance:

symmetric_matrix
----------------

.. automodule:: masspcf.symmetric_matrix
   :members:
   :undoc-members:
   :show-inheritance:

norms
-----

.. automodule:: masspcf.norms
   :members:
   :undoc-members:
   :show-inheritance:

io
--

.. automodule:: masspcf.io
   :members:
   :undoc-members:
   :show-inheritance:

serialize
---------

.. automodule:: masspcf.serialize
   :members:
   :undoc-members:
   :show-inheritance:

plotting
--------

.. automodule:: masspcf.plotting
   :members:
   :undoc-members:
   :show-inheritance:

random
------

.. automodule:: masspcf.random
   :members:
   :undoc-members:
   :show-inheritance:

system
------

The ``masspcf.system`` module provides access to system-wide library settings. Note that these settings are per session and must be reconfigured for each Python kernel run.

Most users should not need to make any changes but we do provide the capability for advanced/expert users. No core functionality in the package requires manual modification of any of these options.

.. automodule:: masspcf.system
   :members:
   :undoc-members:
   :show-inheritance:

gpu
---

.. automodule:: masspcf.gpu
   :members:
   :undoc-members:
   :show-inheritance:

typing
------

.. py:class:: pcf32

   32-bit PCF dtype. Use as the ``dtype`` argument when creating tensors of 32-bit piecewise constant functions.

.. py:class:: pcf64

   64-bit PCF dtype. Use as the ``dtype`` argument when creating tensors of 64-bit piecewise constant functions.

.. py:class:: pcf32i

   32-bit integer PCF dtype. Use as the ``dtype`` argument when creating tensors of 32-bit integer piecewise constant functions.

.. py:class:: pcf64i

   64-bit integer PCF dtype. Use as the ``dtype`` argument when creating tensors of 64-bit integer piecewise constant functions.

.. py:class:: float32

   32-bit floating-point scalar dtype. Use as the ``dtype`` argument when creating tensors of scalar float values.

.. py:class:: float64

   64-bit floating-point scalar dtype. Use as the ``dtype`` argument when creating tensors of scalar float values.

.. py:class:: int32

   32-bit integer scalar dtype. Use as the ``dtype`` argument when creating tensors of scalar integer values.

.. py:class:: int64

   64-bit integer scalar dtype. Use as the ``dtype`` argument when creating tensors of scalar integer values.

.. py:class:: pcloud32

   32-bit point cloud dtype. Use as the ``dtype`` argument when creating tensors of 32-bit point clouds.

.. py:class:: pcloud64

   64-bit point cloud dtype. Use as the ``dtype`` argument when creating tensors of 64-bit point clouds.

.. py:class:: barcode32

   32-bit barcode dtype. Use as the ``dtype`` argument when creating tensors of 32-bit persistence barcodes.

.. py:class:: barcode64

   64-bit barcode dtype. Use as the ``dtype`` argument when creating tensors of 64-bit persistence barcodes.

.. py:class:: symmat32

   32-bit symmetric matrix dtype. Use as the ``dtype`` argument when creating tensors of 32-bit symmetric matrices.

.. py:class:: symmat64

   64-bit symmetric matrix dtype. Use as the ``dtype`` argument when creating tensors of 64-bit symmetric matrices.

