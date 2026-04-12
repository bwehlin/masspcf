====================
Time delay embedding
====================

Time delay embedding (also known as Takens embedding) reconstructs the
phase-space dynamics of a system from a single observed time series.
The function :py:func:`~masspcf.embed_time_delay` converts a ``TimeSeries``
(or ``TimeSeriesTensor``) into a :py:class:`~masspcf.PointCloudTensor`
whose points are delay vectors.

At each valid time *t*, the embedding vector looks backward:

.. math::

   \mathbf{v}(t) = \bigl[\, x(t - (d{-}1)\tau),\; \ldots,\; x(t - \tau),\; x(t) \,\bigr]

where *d* is the embedding ``dimension`` and :math:`\tau` is the ``delay``.
Valid times start at :math:`t_{\min} = t_0 + (d-1)\tau`, where
:math:`t_0` is the start of the series.


Basic usage
-----------

Pass a ``TimeSeries``, the embedding dimension, and the delay::

   import numpy as np
   import masspcf as mpcf

   np.random.seed(42)
   t = np.linspace(0, 4 * np.pi, 200)
   values = np.sin(t) + 0.1 * np.random.randn(len(t))
   ts = mpcf.TimeSeries(values, start_time=0.0, time_step=t[1] - t[0])

   cloud = mpcf.embed_time_delay(ts, dimension=2, delay=0.4)
   cloud.shape   # (1,)
   pts = np.asarray(cloud[0])
   pts.shape     # (n_points, 2)

The result is a :py:class:`~masspcf.PointCloudTensor`. Without windowing
the shape is ``(1,)`` -- a single point cloud. Each row is a delay vector
of length ``dimension``.

.. image:: _static/timeseries_embed_basic_light.png
   :width: 90%
   :class: only-light
   :alt: Time delay embedding of a noisy sinusoid into 2D

.. image:: _static/timeseries_embed_basic_dark.png
   :width: 90%
   :class: only-dark
   :alt: Time delay embedding of a noisy sinusoid into 2D

.. dropdown:: Show plotting code
   :color: secondary

   .. literalinclude:: _static/gen_timeseries_fig.py
      :pyobject: plot_embed_basic
      :language: python

Higher dimensions work the same way -- the point dimension equals ``d``::

   cloud3d = mpcf.embed_time_delay(ts, dimension=3, delay=0.4)
   np.asarray(cloud3d[0]).shape  # (n_points, 3)


Multi-channel embedding
-----------------------

For a multi-channel ``TimeSeries`` with *c* channels, the delay vector
at time *t* interleaves the channels:

.. math::

   \mathbf{v}(t) = \bigl[\,
       x_1(t{-}(d{-}1)\tau),\; x_2(t{-}(d{-}1)\tau),\; \ldots,\;
       x_1(t),\; x_2(t)
   \,\bigr]

so each point has ``dimension * n_channels`` coordinates::

   times = np.arange(5, dtype=np.float64)
   values = np.array([
       [1.0, 2.0, 3.0, 4.0, 5.0],     # channel 0
       [10.0, 20.0, 30.0, 40.0, 50.0], # channel 1
   ])
   ts = mpcf.TimeSeries(times, values)
   cloud = mpcf.embed_time_delay(ts, dimension=2, delay=1.0)
   np.asarray(cloud[0]).shape  # (4, 4)  -- 4 points, each 2*2 = 4D


Windowed embedding
------------------

The ``window`` parameter splits the valid time range into windows of the
given duration. This is useful for detecting changes in dynamics over
time -- each window produces a separate point cloud that can be analyzed
independently::

   ts = mpcf.TimeSeries(np.arange(1.0, 11.0))
   clouds = mpcf.embed_time_delay(
       ts, dimension=2, delay=1.0, window=3.0)
   clouds.shape  # (3,) -- three windows

Windows are anchored at the *end* of the valid range and extend backward,
so the first window may be shorter than ``window``.

Use ``stride`` to control overlap between windows. By default ``stride``
equals ``window`` (non-overlapping). A smaller stride produces overlapping
windows::

   clouds = mpcf.embed_time_delay(
       ts, dimension=2, delay=1.0, window=4.0, stride=2.0)
   clouds.shape  # (3,) -- overlapping windows

.. image:: _static/timeseries_embed_windowed_light.png
   :width: 100%
   :class: only-light
   :alt: Windowed time delay embedding showing per-window point clouds

.. image:: _static/timeseries_embed_windowed_dark.png
   :width: 100%
   :class: only-dark
   :alt: Windowed time delay embedding showing per-window point clouds

.. dropdown:: Show plotting code
   :color: secondary

   .. literalinclude:: _static/gen_timeseries_fig.py
      :pyobject: plot_embed_windowed
      :language: python


Embedding a TimeSeriesTensor
----------------------------

When applied to a ``TimeSeriesTensor`` of shape *S*, the result is a
``PointCloudTensor`` of shape *S* (without windowing) or
*S* + ``(n_windows,)`` (with windowing). Each series is embedded
independently::

   ts1 = mpcf.TimeSeries(np.arange(10, dtype=np.float64))
   ts2 = mpcf.TimeSeries(np.arange(10, dtype=np.float64) * 10)
   tensor = mpcf.TimeSeriesTensor([ts1, ts2])

   clouds = mpcf.embed_time_delay(tensor, dimension=2, delay=1.0)
   clouds.shape  # (2,)

   windowed = mpcf.embed_time_delay(
       tensor, dimension=2, delay=1.0, window=3.0)
   windowed.shape  # (2, n_windows)


Datetime delays
---------------

When the time series uses ``datetime64`` times, pass the delay as a
``timedelta64``::

   times = np.array([
       "2024-01-01T00:00:00",
       "2024-01-01T00:00:01",
       "2024-01-01T00:00:02",
       "2024-01-01T00:00:03",
       "2024-01-01T00:00:04",
   ], dtype="datetime64[s]")
   ts = mpcf.TimeSeries(times, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
   cloud = mpcf.embed_time_delay(
       ts, dimension=2, delay=np.timedelta64(1, 's'))


Connecting to TDA
-----------------

The point clouds produced by ``embed_time_delay`` integrate naturally
with masspcf's persistence pipeline. A typical workflow:

1. Embed a time series into a point cloud.
2. Compute persistent homology with
   :py:func:`~masspcf.persistence.compute_persistent_homology`.
3. Summarize with
   :py:func:`~masspcf.persistence.barcode_to_stable_rank` or
   :py:func:`~masspcf.persistence.barcode_to_betti_curve`.

::

   from masspcf.persistence import (
       compute_persistent_homology,
       barcode_to_stable_rank,
   )

   cloud = mpcf.embed_time_delay(ts, dimension=3, delay=0.5)
   barcodes = compute_persistent_homology(cloud, max_dim=1)
   sr = barcode_to_stable_rank(barcodes)

For a complete worked example -- including windowed embedding and
topological regime-change detection -- see the
:doc:`Lorenz attractor tutorial <tutorial_notebooks/lorenz_takens_embedding>`.
