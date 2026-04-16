==========================
Creating a time series
==========================

A :py:class:`~masspcf.TimeSeries` wraps a piecewise constant function with
real-world time metadata, making it easy to work with sensor readings,
sampled signals, and other time-stamped data in the masspcf framework.

Evaluating outside the series domain (before the first sample or after the
last) returns ``NaN``.


From times and values
---------------------

Pass separate arrays of sample times and values::

   import numpy as np
   import masspcf as mpcf

   times = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
   values = np.array([1.0, 3.0, 2.0, 4.0, 1.5])
   ts = mpcf.TimeSeries(times, values)
   ts.start_time  # 10.0

.. image:: _static/timeseries_basic_light.png
   :width: 80%
   :class: only-light
   :alt: A time series created from explicit times and values

.. image:: _static/timeseries_basic_dark.png
   :width: 80%
   :class: only-dark
   :alt: A time series created from explicit times and values

.. dropdown:: Show plotting code
   :color: secondary

   .. literalinclude:: _static/gen_timeseries_fig.py
      :pyobject: plot_timeseries_basic
      :language: python


From regularly-sampled values
-----------------------------

When samples are equally spaced, pass just the values with
``start_time`` and ``time_step``::

   ts = mpcf.TimeSeries(
       np.array([1.0, 3.0, 2.0, 4.0, 1.5]),
       start_time=10.0,
       time_step=2.0,
   )
   ts.times    # array([10., 12., 14., 16., 18.])
   ts.values   # array([1. , 3. , 2. , 4. , 1.5])

When using numeric (float) times, the default ``start_time`` is ``0.0``
and the default ``time_step`` is ``1.0``. Internally, float times are
interpreted as seconds since the Unix epoch (1970-01-01T00:00:00), which
is also how ``datetime64`` times are represented under the hood.

Datetime support
----------------

Both construction forms accept ``datetime64`` times. All numpy datetime64
units are supported, from attoseconds (``as``) through years (``Y``).
Pass ``datetime64`` arrays directly, or use ``start_time`` / ``time_step``
with ``datetime64`` / ``timedelta64``::

   # From datetime arrays
   times = np.array([
       "2024-06-15T08:00:00.000",
       "2024-06-15T08:00:00.010",
       "2024-06-15T08:00:00.020",
       "2024-06-15T08:00:00.030",
   ], dtype="datetime64[ms]")
   ts = mpcf.TimeSeries(times, np.array([22.1, 22.3, 23.0, 22.8]))

   # Or from regularly-sampled values
   ts = mpcf.TimeSeries(
       np.array([22.1, 22.3, 23.0, 22.8]),
       start_time=np.datetime64("2024-06-15T08:00:00"),
       time_step=np.timedelta64(10, "ms"),
   )

Query times can also be ``datetime64``::

   ts(np.datetime64("2024-06-15T08:00:00.015"))  # 23.0

Here is a more complete example with two datetime-based sensors that start at
different times and sample at different rates::

   epoch1 = np.datetime64("2024-06-15T08:00:00")
   epoch2 = np.datetime64("2024-06-15T08:00:02")

   ts1 = mpcf.TimeSeries(
       np.array([22.1, 22.3, 23.0, 22.8, 22.5, 23.2, 24.0, 23.5]),
       start_time=epoch1,
       time_step=np.timedelta64(500, "ms"),
   )
   ts2 = mpcf.TimeSeries(
       np.array([21.0, 21.8, 22.5, 23.1, 22.9]),
       start_time=epoch2,
       time_step=np.timedelta64(1, "s"),
   )

   # Query both at the same wall-clock time
   t = np.datetime64("2024-06-15T08:00:02.700")
   ts1(t)  # 23.2
   ts2(t)  # 21.0

.. image:: _static/timeseries_datetime_light.png
   :width: 90%
   :class: only-light
   :alt: Two datetime time series with different start times and sampling rates

.. image:: _static/timeseries_datetime_dark.png
   :width: 90%
   :class: only-dark
   :alt: Two datetime time series with different start times and sampling rates

.. dropdown:: Show plotting code
   :color: secondary

   .. literalinclude:: _static/gen_timeseries_fig.py
      :pyobject: plot_datetime_example
      :language: python

Internally, all datetime values are converted to seconds since the Unix
epoch (1970-01-01T00:00:00). The ``start_time`` and ``times`` properties
return these float seconds. Fixed-length units (``as`` through ``W``) are
converted exactly via ``std::chrono``. For variable-length units (``M``
and ``Y``), numpy's calendar-aware arithmetic computes exact dates, which
are then converted to float seconds (no approximation).

.. note::

   **Precision.** Times are stored as float64 seconds. Float64 has
   about 15--16 significant digits, so precision degrades for times far
   from zero. Near the current epoch (~1.7 × 10\ :sup:`9` s), the
   smallest representable difference is roughly 200 ns. Near time zero,
   the full precision of the input unit is preserved.

   **Timezones.** All ``datetime64`` values must be timezone-naive
   (numpy's ``datetime64`` does not carry timezone information). If you
   have timezone-aware timestamps, convert to UTC and strip the timezone
   before constructing a ``TimeSeries``.


Multi-channel time series
=========================

When the data has multiple channels (e.g. a 3-axis accelerometer or
multi-sensor readings), pass a 2-D values array of shape
``(n_times, n_channels)``::

   # 2 channels (e.g. temperature + humidity), 3 time points
   times = np.array([0.0, 1.0, 2.0])
   values = np.array([
       [22.1, 45.0],   # t=0: temperature, humidity
       [22.5, 44.0],   # t=1
       [23.0, 43.5],   # t=2
   ])
   ts = mpcf.TimeSeries(times, values)
   ts.n_channels  # 2

Evaluating at a single time returns one value per channel::

   ts(0.5)  # array([22.1, 45.0])

Evaluating at multiple times returns shape ``(n_times, n_channels)``::

   ts(np.array([0.5, 1.5]))
   # array([[22.1, 45.0],    # t=0.5: temperature, humidity
   #        [22.5, 44.0]])   # t=1.5

The regularly-sampled form also accepts 2-D values::

   values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
   ts = mpcf.TimeSeries(values, start_time=0.0, time_step=0.5)
   ts.n_channels  # 2

Single-channel time series (1-D values) behave exactly as before --
scalar evaluation returns a float, array evaluation returns a 1-D array.

.. image:: _static/timeseries_multichannel_light.png
   :width: 90%
   :class: only-light
   :alt: Multi-channel time series with temperature and humidity channels

.. image:: _static/timeseries_multichannel_dark.png
   :width: 90%
   :class: only-dark
   :alt: Multi-channel time series with temperature and humidity channels

.. dropdown:: Show plotting code
   :color: secondary

   .. literalinclude:: _static/gen_timeseries_fig.py
      :pyobject: plot_multichannel
      :language: python


Evaluation
==========

Time series are callable. Pass a single time or an array of times::

   ts = mpcf.TimeSeries(np.array([1.0, 3.0, 2.0, 4.0, 1.5]),
                         start_time=10.0, time_step=2.0)

   ts(10.0)                       # 1.0
   ts(13.0)                       # 3.0
   ts(np.array([10.0, 14.0]))     # array([1.0, 2.0])

Times before the start or after the last sample return ``NaN``::

   ts(9.0)    # nan
   ts(20.0)   # nan

.. image:: _static/timeseries_eval_light.png
   :width: 80%
   :class: only-light
   :alt: Time series evaluation with NaN for out-of-range times

.. image:: _static/timeseries_eval_dark.png
   :width: 80%
   :class: only-dark
   :alt: Time series evaluation with NaN for out-of-range times

.. dropdown:: Show plotting code
   :color: secondary

   .. literalinclude:: _static/gen_timeseries_fig.py
      :pyobject: plot_timeseries_eval
      :language: python


Interpolation
=============

By default, a ``TimeSeries`` evaluates as a piecewise constant (step)
function: querying between breakpoints returns the value of the left
breakpoint.  The ``interpolation`` parameter controls this behavior:

``'nearest'`` (default)
   Step function -- each sample holds its value until the next sample.

``'linear'``
   Linear interpolation between adjacent samples.

Set the mode at construction time or change it later::

   ts = mpcf.TimeSeries(
       np.array([1.0, 3.0, 2.0, 4.0, 1.5]),
       start_time=10.0, time_step=2.0,
       interpolation='linear',
   )
   ts(13.0)  # 2.5 (halfway between 3.0 and 2.0)

   # Change mode on an existing series
   ts.interpolation = 'nearest'
   ts(13.0)  # 3.0

.. image:: _static/timeseries_interpolation_light.png
   :width: 90%
   :class: only-light
   :alt: Comparison of nearest and linear interpolation modes

.. image:: _static/timeseries_interpolation_dark.png
   :width: 90%
   :class: only-dark
   :alt: Comparison of nearest and linear interpolation modes

.. dropdown:: Show plotting code
   :color: secondary

   .. literalinclude:: _static/gen_timeseries_fig.py
      :pyobject: plot_interpolation
      :language: python

The interpolation mode is a property of the ``TimeSeries`` object, so
functions like :py:func:`~masspcf.embed_time_delay` automatically
respect it.  For example, a linear-interpolation embedding produces
smoother point clouds than a nearest-neighbor one.

Multi-channel time series interpolate each channel independently.


Custom interpolation strategies
-------------------------------

When ``'nearest'`` and ``'linear'`` are not enough -- optimal transport
between images, cubic splines, or any other scheme -- supply a Python
callable via :py:class:`~masspcf.timeseries.CallableInterpolation`. The
callable runs once per query batch; between calls the time series
pre-computes the bracketing breakpoints, so you only have to implement
the interpolation itself::

   from masspcf.timeseries import CallableInterpolation
   import numpy as np

   def step_from_right(queries, t_lefts, t_rights, v_lefts, v_rights):
       # Return one interpolated value per query (same length as queries).
       return list(v_rights)

   ts = mpcf.TimeSeries(
       np.array([1.0, 2.0, 3.0]),
       start_time=0.0, time_step=1.0,
       interpolation=CallableInterpolation(step_from_right),
   )

The callback signature is always batched -- you receive five parallel
sequences of length *n_queries* and return *n_queries* interpolated
values:

``queries``
   Query times in the internal PCF parametrization (float).

``t_lefts``, ``t_rights``
   Times of the bracketing left/right breakpoints. At the right domain
   boundary, ``t_rights[i] == t_lefts[i]``; treat that as "no right
   neighbor" and return ``v_lefts[i]``.

``v_lefts``, ``v_rights``
   Values at the bracketing breakpoints. For scalar series these are
   sequences of floats; for PCF- or barcode-valued series (below), they
   are sequences of :py:class:`~masspcf.Pcf` / :py:class:`~masspcf.Barcode`
   wrapper objects.

Custom strategies are shared on the C++ side, so a single instance --
potentially holding expensive state like a trained OT model -- can be
attached to many time series::

   ot_interp = CallableInterpolation(my_ot_model.interpolate)
   ts_a = mpcf.TimeSeries(times_a, imgs_a, interpolation=ot_interp)
   ts_b = mpcf.TimeSeries(times_b, imgs_b, interpolation=ot_interp)

The ``interpolation`` property reports ``'custom'`` whenever a callable
strategy is attached.

.. note::

   Python callbacks acquire the GIL once per batch, so custom strategies
   serialize across threads. Built-in ``'nearest'`` and ``'linear'`` run
   inline in C++ and are unaffected.


Time series of PCFs and barcodes
================================

Each sample in a time series does not have to be a number. Passing a
list of :py:class:`~masspcf.Pcf` or :py:class:`~masspcf.Barcode` objects
as the values produces a time series whose values evolve over time::

   from masspcf import Pcf, TimeSeries
   import numpy as np

   p1 = Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]], dtype=np.float32))
   p2 = Pcf(np.array([[0.0, 3.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32))

   # One PCF per time point
   ts = TimeSeries([0.0, 1.0], [p1, p2])

   ts(0.0)   # returns p1 (a Pcf)
   ts(0.3)   # nearest: returns p1
   ts.dtype  # masspcf.pcf32 — the per-element dtype

Evaluation returns a :py:class:`~masspcf.Pcf` (or
:py:class:`~masspcf.Barcode`, etc.) wrapper rather than a scalar.

Interpolation for PCFs
----------------------

PCFs support arithmetic (addition and scalar multiplication), so
``'linear'`` interpolation is well defined and produces a pointwise blend::

   ts = TimeSeries([0.0, 1.0], [p1, p2], interpolation='linear')
   blended = ts(0.5)
   blended(0.5)  # 0.5 * p1(0.5) + 0.5 * p2(0.5)

Interpolation for barcodes
--------------------------

There is no canonical way to linearly interpolate two persistence
barcodes, so only ``'nearest'`` is built in. Requesting ``'linear'`` on a
barcode-valued time series raises::

   from masspcf.persistence import Barcode

   b1 = Barcode(np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float32))
   b2 = Barcode(np.array([[0.0, 2.0], [0.3, 3.0]], dtype=np.float32))

   ts = TimeSeries([0.0, 1.0], [b1, b2])  # nearest is fine
   ts(0.3)  # Barcode, equal to b1

If a domain-specific interpolation scheme for barcodes is needed,
supply a :py:class:`~masspcf.timeseries.CallableInterpolation` that
implements it.

Limitations of the initial version
----------------------------------

* The object-valued path always has ``n_channels == 1``.
* :py:func:`~masspcf.embed_time_delay` is defined only for scalar time
  series. Embedding a time series of PCFs or barcodes is a planned
  extension.
* Pickling / ``save`` and ``load`` are only implemented for scalar
  series in this release.
* ``datetime64`` time axes are only implemented for scalar series so
  far. Float time axes work for all value types.


For per-time-step **tensor** values (point clouds, images, tensors of
PCFs or barcodes) use :py:class:`~masspcf.TensorTimeSeries` — covered
in :doc:`ts_tensor`.


TimeSeriesTensor
================

A :py:class:`~masspcf.TimeSeriesTensor` holds a collection of time series.
Each series can have its own start time and sampling rate::

   # Fast sensor: 0.5s intervals, starts at t=1
   fast = mpcf.TimeSeries(
       np.array([2.1, 2.5, 3.0, 2.8, 2.3, 2.9, 3.2, 2.7, 2.4, 3.1]),
       start_time=1.0, time_step=0.5,
   )
   # Slow sensor: 1.5s intervals, starts at t=0
   slow = mpcf.TimeSeries(
       np.array([10.0, 12.0, 11.5, 13.0, 12.5]),
       start_time=0.0, time_step=1.5,
   )
   tensor = mpcf.TimeSeriesTensor([fast, slow])

   # Both evaluated at the same real time
   tensor(3.5)   # array([2.9, 12.0])

.. image:: _static/timeseries_different_scales_light.png
   :width: 90%
   :class: only-light
   :alt: Two sensors with different sampling rates evaluated at the same time

.. image:: _static/timeseries_different_scales_dark.png
   :width: 90%
   :class: only-dark
   :alt: Two sensors with different sampling rates evaluated at the same time

.. dropdown:: Show plotting code
   :color: secondary

   .. literalinclude:: _static/gen_timeseries_fig.py
      :pyobject: plot_different_scales
      :language: python

Evaluating a tensor queries every series at the given time. Series
where the query falls outside their domain return ``NaN``::

   tensor(0.5)
   # array([ 2.1,  nan])
   # fast: t=0.5 is before start (1.0) -> NaN... wait, 0.5 < 1.0 -> NaN
   # slow: t=0.5 is in [0, 1.5) -> 10.0

Array evaluation appends the time dimensions to the tensor shape, just like
PCF tensor evaluation::

   tensor(np.array([1.5, 3.5]))
   # shape (2, 2) -- tensor shape (2,) + times shape (2,)

The ``start_times`` and ``end_times`` properties give the time domain
of each series::

   tensor.start_times  # array([1. , 0. ])
   tensor.end_times    # array([5.5, 6. ])


Dtypes
======

``TimeSeries.dtype`` reports the **per-element** dtype — the same dtype
you'd see on a standalone sample value. Scalar series carry
``float32`` / ``float64``; PCF-valued series carry ``pcf32`` / ``pcf64``;
barcode-valued series carry ``barcode32`` / ``barcode64``::

   ts = mpcf.TimeSeries(np.array([1.0], dtype=np.float32),
                         start_time=0.0, time_step=1.0)
   ts.dtype  # masspcf.float32

   ts = mpcf.TimeSeries(np.array([1.0], dtype=np.float64),
                         start_time=0.0, time_step=1.0)
   ts.dtype  # masspcf.float64

Use ``float32`` for lower memory usage, ``float64`` (the default) for
higher precision.

For object-valued series (PCFs, barcodes), the dtype is inferred from
the first element, so there is usually nothing explicit to pass.
