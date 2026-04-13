================
Interoperability
================

The ``TimeSeries`` constructor accepts NumPy arrays, so converting from
common data-science libraries is straightforward.


From a pandas Series
--------------------

A ``pandas.Series`` with a numeric or ``DatetimeIndex`` converts
directly::

   import pandas as pd
   import numpy as np
   import masspcf as mpcf

   # Numeric index
   s = pd.Series([1.0, 3.0, 2.0, 4.0], index=[0.0, 0.5, 1.0, 1.5])
   ts = mpcf.TimeSeries(s.index.to_numpy(), s.to_numpy())

   # DatetimeIndex -- pass the index as times
   idx = pd.date_range("2024-01-01", periods=100, freq="h")
   s = pd.Series(np.random.randn(100), index=idx)
   ts = mpcf.TimeSeries(s.index.to_numpy(), s.to_numpy())

This works for both regularly and irregularly sampled series -- the
``TimeSeries`` constructor infers the time step from the provided
times.


From a pandas DataFrame
-----------------------

Each column of a ``DataFrame`` can become a separate ``TimeSeries``.
Use a ``TimeSeriesTensor`` to group them::

   df = pd.DataFrame({
       "sensor_a": np.random.randn(100),
       "sensor_b": np.random.randn(100),
       "sensor_c": np.random.randn(100),
   }, index=pd.date_range("2024-01-01", periods=100, freq="min"))

   series = [
       mpcf.TimeSeries(df.index.to_numpy(), df[col].to_numpy())
       for col in df.columns
   ]
   tensor = mpcf.TimeSeriesTensor(series)

Alternatively, if the columns represent channels of the *same* signal,
pass the full array as a multi-channel ``TimeSeries``. The DataFrame's
natural ``(n_times, n_columns)`` layout matches directly::

   ts = mpcf.TimeSeries(
       df.index.to_numpy(),
       df.to_numpy(),   # (n_times, n_channels) -- no transpose needed
   )
   ts.n_channels  # 3


From a CSV file
---------------

Use pandas (or any CSV reader) to load the file, then convert::

   df = pd.read_csv("sensor_log.csv", parse_dates=["timestamp"])
   ts = mpcf.TimeSeries(
       df["timestamp"].to_numpy().astype("datetime64[s]"),
       df["value"].to_numpy(),
   )


From an xarray DataArray
------------------------

``xarray`` stores coordinates as NumPy arrays under the hood::

   import xarray as xr

   # 1-D DataArray with a time coordinate
   da = xr.DataArray(
       np.random.randn(200),
       dims=["time"],
       coords={"time": pd.date_range("2024-06-01", periods=200, freq="h")},
   )
   ts = mpcf.TimeSeries(
       da.coords["time"].values,   # datetime64 array
       da.values,
   )


From a polars DataFrame
-----------------------

Polars columns convert to NumPy via ``.to_numpy()``::

   import polars as pl

   df = pl.DataFrame({
       "timestamp": pl.datetime_range(
           datetime(2024, 1, 1), datetime(2024, 1, 2),
           interval="1h", eager=True,
       ),
       "value": np.random.randn(25),
   })
   ts = mpcf.TimeSeries(
       df["timestamp"].to_numpy().astype("datetime64[us]"),
       df["value"].to_numpy(),
   )


From an sktime dataset
----------------------

`sktime <https://www.sktime.net>`_ datasets are typically ``pandas.Series``
objects with a ``PeriodIndex``. Convert the index to timestamps first::

   from sktime.datasets import load_airline

   y = load_airline()                          # Series with PeriodIndex
   times = y.index.to_timestamp().to_numpy()   # -> datetime64 array
   ts = mpcf.TimeSeries(times, y.to_numpy().astype(float))

For a complete classification example using sktime's multi-channel
BasicMotions dataset, see the
:doc:`motion classification tutorial <tutorial_notebooks/motion_classification>`.
