===========
Time Series
===========

A :py:class:`~masspcf.TimeSeries` wraps a piecewise constant function with
real-world time metadata, making it easy to work with sensor readings,
sampled signals, and other time-stamped data in the masspcf framework.

Evaluating outside the series domain (before the first sample or after the
last) returns ``NaN``.


Creating a time series
======================

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

Datetime support
----------------

Both construction forms accept ``datetime64`` times. Pass ``datetime64``
arrays directly, or use ``start_time`` / ``time_step`` with
``datetime64`` / ``timedelta64``::

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

Time series tensors use the ``ts32`` and ``ts64`` dtypes::

   ts = mpcf.TimeSeries(np.array([1.0], dtype=np.float32),
                         start_time=0.0, time_step=1.0)
   ts.dtype  # masspcf.ts32

   ts = mpcf.TimeSeries(np.array([1.0], dtype=np.float64),
                         start_time=0.0, time_step=1.0)
   ts.dtype  # masspcf.ts64

Use ``ts32`` for lower memory usage, ``ts64`` (the default) for higher
precision.
