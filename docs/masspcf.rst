masspcf Python API reference
============================

tensor
------------------

.. automodule:: masspcf.tensor
   :members:
   :undoc-members:
   :show-inheritance:

tensor_create
---------------------

.. automodule:: masspcf.tensor_create
   :members:
   :undoc-members:
   :show-inheritance:

pcf
------------------

.. automodule:: masspcf.pcf
   :members:
   :undoc-members:
   :show-inheritance:

reductions
---------------------

.. automodule:: masspcf.reductions
   :members:
   :undoc-members:
   :show-inheritance:

distance
---------------------

.. automodule:: masspcf.distance
   :members:
   :undoc-members:
   :show-inheritance:

norms
------------------

.. automodule:: masspcf.norms
   :members:
   :undoc-members:
   :show-inheritance:

io
---------------------

.. automodule:: masspcf.io
   :members:
   :undoc-members:
   :show-inheritance:

serialize
---------------------

.. automodule:: masspcf.serialize
   :members:
   :undoc-members:
   :show-inheritance:

plotting
-----------------------

.. automodule:: masspcf.plotting
   :members:
   :undoc-members:
   :show-inheritance:

random
---------------------

.. automodule:: masspcf.random
   :members:
   :undoc-members:
   :show-inheritance:

system
---------------------

The `masspcf.system` module provides access to system-wide library settings. Note that these settings are per session and must be reconfigured for each Python kernel run.

Most users should not need to make any changes but we do provide the capability for advanced/expert users. No core functionality in the package requires manual modification of any of these options.

.. automodule:: masspcf.system
   :members:
   :undoc-members:
   :show-inheritance:

gpu
---------------------

.. automodule:: masspcf.gpu
   :members:
   :undoc-members:
   :show-inheritance:

persistence
---------------------

.. automodule:: masspcf.persistence
   :members:
   :undoc-members:
   :show-inheritance:

persistence.barcode
~~~~~~~~~~~~~~~~~~~

.. automodule:: masspcf.persistence.barcode
   :members:
   :undoc-members:
   :show-inheritance:

persistence.ph_tensor
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: masspcf.persistence.ph_tensor
   :members:
   :undoc-members:
   :show-inheritance:

persistence.homology
~~~~~~~~~~~~~~~~~~~~

.. automodule:: masspcf.persistence.homology
   :members:
   :undoc-members:
   :show-inheritance:

persistence.stable_rank
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: masspcf.persistence.stable_rank
   :members:
   :undoc-members:
   :show-inheritance:

typing
---------------------

.. py:class:: pcf32

   32-bit PCF dtype. Use as the ``dtype`` argument when creating tensors of 32-bit piecewise constant functions.

.. py:class:: pcf64

   64-bit PCF dtype. Use as the ``dtype`` argument when creating tensors of 64-bit piecewise constant functions.

.. py:data:: f32

   32-bit floating-point scalar dtype (alias for ``numpy.float32``). Use as the ``dtype`` argument when creating tensors of scalar float values.

.. py:data:: f64

   64-bit floating-point scalar dtype (alias for ``numpy.float64``). Use as the ``dtype`` argument when creating tensors of scalar float values.

.. py:class:: pcloud32

   32-bit point cloud dtype. Use as the ``dtype`` argument when creating tensors of 32-bit point clouds.

.. py:class:: pcloud64

   64-bit point cloud dtype. Use as the ``dtype`` argument when creating tensors of 64-bit point clouds.

.. py:class:: barcode32

   32-bit barcode dtype. Use as the ``dtype`` argument when creating tensors of 32-bit persistence barcodes.

.. py:class:: barcode64

   64-bit barcode dtype. Use as the ``dtype`` argument when creating tensors of 64-bit persistence barcodes.

.. py:class:: float32

   .. deprecated::
      Use :py:class:`pcf32` or :py:data:`f32` instead.

   32-bit floating-point type (type alias for `numpy.float32` -- see `numpy` documentation for more information.)

.. py:class:: float64

   .. deprecated::
      Use :py:class:`pcf64` or :py:data:`f64` instead.

   64-bit floating-point type (type alias for `numpy.float64` -- see `numpy` documentation for more information.)

