======================
Time series of tensors
======================

When each time step is a whole tensor — a point cloud, an image, a
tensor of PCFs, a tensor of barcodes — use
:py:class:`~masspcf.TensorTimeSeries` instead of
:py:class:`~masspcf.TimeSeries`. Same API (callable, ``interpolation``,
``times``, ``values``), but each sample is a
:py:class:`~masspcf.FloatTensor`, :py:class:`~masspcf.PcfTensor`, or
:py:class:`~masspcf.BarcodeTensor`.


Point cloud per time step
=========================

The most common use case is a time-varying point cloud::

   import numpy as np
   from masspcf import TensorTimeSeries, FloatTensor

   rng = np.random.default_rng(0)
   clouds = [FloatTensor(rng.standard_normal((12, 2)).astype(np.float32))
             for _ in range(5)]

   pc_ts = TensorTimeSeries([0.0, 1.0, 2.0, 3.0, 4.0], clouds)
   pc_ts.dtype   # masspcf.float32  (the dtype of the tensor's elements)
   pc_ts(2.0)    # FloatTensor — nearest picks the sample at t=2

Numpy arrays in the values list are accepted directly and coerced into
``FloatTensor`` automatically. Clouds at different time steps may have
different point counts.


Interpolation
=============

Linear interpolation is explicitly disabled for tensor-valued series
because blending two tensors with different shapes is undefined.
Requesting ``'linear'`` on a ``TensorTimeSeries`` raises::

   pc_ts.interpolation = 'linear'   # TypeError: linear interpolation ...

If your data has a consistent shape and a meaningful blend, supply a
:py:class:`~masspcf.timeseries.CallableInterpolation`. See
:doc:`ts_creating` for details on custom strategies.


Persistent homology over time
=============================

Feeding a point-cloud ``TensorTimeSeries`` to
:py:func:`~masspcf.persistence.compute_persistent_homology` computes a
barcode per time step and returns a new ``TensorTimeSeries`` in which
each sample is a :py:class:`~masspcf.BarcodeTensor` indexed by homology
dimension::

   from masspcf.persistence import compute_persistent_homology

   bc_ts = compute_persistent_homology(pc_ts, max_dim=1)
   bc_ts.dtype   # masspcf.barcode32
   bc_ts(2.0)    # BarcodeTensor of shape (2,)  — [H0, H1] at t=2

   bc_ts(2.0)[0]  # H0 barcode at t=2
   bc_ts(2.0)[1]  # H1 barcode at t=2

The time axis, ``start_time``, and ``time_step`` are preserved from the
input. The output always uses nearest interpolation (barcodes have no
canonical linear blend, and custom strategies don't port between
element types).


Other tensor element types
==========================

``TensorTimeSeries`` also supports :py:class:`~masspcf.PcfTensor` and
:py:class:`~masspcf.BarcodeTensor` per time step — useful when a
pipeline produces several PCFs or barcodes at every step::

   from masspcf import PcfTensor
   pcftensor_ts = TensorTimeSeries(times, list_of_pcftensors)
   pcftensor_ts.dtype  # masspcf.pcf32


Dtype convention
================

``TensorTimeSeries.dtype`` reports the dtype of the **scalar elements
inside** each per-time-step tensor, not a separate time-series dtype.
This matches the dtype you'd see on a bare tensor of the same kind:

=============================  ====================
Per-time-step tensor class     ``.dtype``
=============================  ====================
:class:`FloatTensor` (f32)     ``float32``
:class:`FloatTensor` (f64)     ``float64``
:class:`PcfTensor` (pcf32)     ``pcf32``
:class:`PcfTensor` (pcf64)     ``pcf64``
:class:`BarcodeTensor` (b32)   ``barcode32``
:class:`BarcodeTensor` (b64)   ``barcode64``
=============================  ====================

The class (:class:`TimeSeries` vs :class:`TensorTimeSeries`) carries
the per-time-step *structure* — one value vs. one tensor — while the
dtype carries the element *precision*.
