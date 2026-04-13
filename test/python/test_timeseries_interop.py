"""Tests for TimeSeries interoperability with pandas, xarray, polars, and sktime."""

from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
from sktime.datasets import load_airline

import masspcf as mpcf


# ---------------------------------------------------------------------------
# pandas
# ---------------------------------------------------------------------------


class TestPandasSeries:
    def test_numeric_index(self):
        s = pd.Series([1.0, 3.0, 2.0, 4.0], index=[0.0, 0.5, 1.0, 1.5])
        ts = mpcf.TimeSeries(s.index.to_numpy(), s.to_numpy())

        assert ts.n_times == 4
        assert ts(0.0) == pytest.approx(1.0)
        assert ts(0.5) == pytest.approx(3.0)

    def test_datetime_index_regular(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        s = pd.Series(values, index=idx)

        ts = mpcf.TimeSeries(s.index.to_numpy(), s.to_numpy())

        assert ts.n_times == 5
        np.testing.assert_array_equal(ts.values, values)

    def test_datetime_index_irregular(self):
        times = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-07",
                                "2024-01-10"])
        values = np.array([1.0, 2.0, 3.0, 4.0])
        s = pd.Series(values, index=times)

        ts = mpcf.TimeSeries(s.index.to_numpy(), s.to_numpy())

        assert ts.n_times == 4
        t = np.datetime64("2024-01-03")
        assert ts(t) == pytest.approx(2.0)

    def test_datetime_index_evaluation_roundtrip(self):
        """Values survive the pandas -> TimeSeries conversion."""
        idx = pd.date_range("2024-06-15T08:00", periods=10, freq="s")
        values = np.arange(10, dtype=np.float64)
        s = pd.Series(values, index=idx)

        ts = mpcf.TimeSeries(s.index.to_numpy(), s.to_numpy())

        result = ts(np.array([
            np.datetime64("2024-06-15T08:00:00"),
            np.datetime64("2024-06-15T08:00:05"),
        ]))
        np.testing.assert_array_almost_equal(result, [0.0, 5.0])


class TestPandasDataFrame:
    def test_columns_to_tensor(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        }, index=pd.date_range("2024-01-01", periods=4, freq="D"))

        series = [
            mpcf.TimeSeries(df.index.to_numpy(), df[col].to_numpy())
            for col in df.columns
        ]
        tensor = mpcf.TimeSeriesTensor(series)

        assert tensor.shape == (2,)
        result = tensor(np.datetime64("2024-01-01"))
        np.testing.assert_array_almost_equal(result, [1.0, 10.0])

    def test_dataframe_to_multichannel(self):
        df = pd.DataFrame({
            "temp": [22.1, 22.5, 23.0],
            "humidity": [45.0, 44.0, 43.5],
        }, index=pd.date_range("2024-01-01", periods=3, freq="h"))

        ts = mpcf.TimeSeries(
            df.index.to_numpy(),
            df.to_numpy(),   # (n_times, n_channels) -- no transpose needed
        )

        assert ts.n_channels == 2
        assert ts.n_times == 3


class TestPandasCSV:
    def test_csv_roundtrip(self, tmp_path):
        """Simulate loading a CSV via pandas."""
        csv_path = tmp_path / "sensor.csv"
        csv_path.write_text(
            "timestamp,value\n"
            "2024-01-01T00:00:00,1.0\n"
            "2024-01-01T01:00:00,2.0\n"
            "2024-01-01T02:00:00,3.0\n"
            "2024-01-01T03:00:00,4.0\n"
        )

        df = pd.read_csv(str(csv_path), parse_dates=["timestamp"])
        ts = mpcf.TimeSeries(
            df["timestamp"].to_numpy().astype("datetime64[s]"),
            df["value"].to_numpy(),
        )

        assert ts.n_times == 4
        assert ts(np.datetime64("2024-01-01T01:00:00")) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# xarray
# ---------------------------------------------------------------------------


class TestXarray:
    def test_dataarray_datetime(self):
        da = xr.DataArray(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            dims=["time"],
            coords={"time": pd.date_range("2024-06-01", periods=5, freq="h")},
        )

        ts = mpcf.TimeSeries(
            da.coords["time"].values,
            da.values,
        )

        assert ts.n_times == 5
        assert ts(np.datetime64("2024-06-01T02:00")) == pytest.approx(3.0)

    def test_dataarray_numeric(self):
        da = xr.DataArray(
            np.array([10.0, 20.0, 30.0]),
            dims=["time"],
            coords={"time": np.array([0.0, 1.0, 2.0])},
        )

        ts = mpcf.TimeSeries(
            da.coords["time"].values,
            da.values,
        )

        assert ts.n_times == 3
        assert ts(1.0) == pytest.approx(20.0)

    def test_dataset_multiple_vars(self):
        ds = xr.Dataset(
            {
                "temp": (["time"], np.array([22.0, 23.0, 24.0])),
                "pressure": (["time"], np.array([1013.0, 1012.0, 1011.0])),
            },
            coords={"time": pd.date_range("2024-01-01", periods=3, freq="D")},
        )

        series = [
            mpcf.TimeSeries(
                ds.coords["time"].values,
                ds[var].values,
            )
            for var in ds.data_vars
        ]
        tensor = mpcf.TimeSeriesTensor(series)

        assert tensor.shape == (2,)


# ---------------------------------------------------------------------------
# polars
# ---------------------------------------------------------------------------


class TestPolars:
    def test_numeric(self):
        df = pl.DataFrame({
            "time": [0.0, 1.0, 2.0, 3.0],
            "value": [10.0, 20.0, 30.0, 40.0],
        })

        ts = mpcf.TimeSeries(
            df["time"].to_numpy(),
            df["value"].to_numpy(),
        )

        assert ts.n_times == 4
        assert ts(1.0) == pytest.approx(20.0)

    def test_datetime(self):
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 0),
                datetime(2024, 1, 1, 1),
                datetime(2024, 1, 1, 2),
                datetime(2024, 1, 1, 3),
            ],
            "value": [1.0, 2.0, 3.0, 4.0],
        })

        times = df["timestamp"].to_numpy().astype("datetime64[us]")
        ts = mpcf.TimeSeries(times, df["value"].to_numpy())

        assert ts.n_times == 4
        assert ts(np.datetime64("2024-01-01T01:00:00")) == pytest.approx(2.0)

    def test_multiple_columns_to_tensor(self):
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        })

        series = [
            mpcf.TimeSeries(df[col].to_numpy())
            for col in df.columns
        ]
        tensor = mpcf.TimeSeriesTensor(series)

        assert tensor.shape == (2,)


# ---------------------------------------------------------------------------
# sktime
# ---------------------------------------------------------------------------


class TestSktime:
    def test_load_airline_dataset(self):
        """Load the classic airline passengers dataset from sktime."""
        y = load_airline()  # pd.Series with PeriodIndex

        # Convert PeriodIndex to datetime64
        times = y.index.to_timestamp().to_numpy()
        ts = mpcf.TimeSeries(times, y.to_numpy().astype(np.float64))

        assert ts.n_times == len(y)

    def test_sktime_series_to_tensor(self):
        """Convert an sktime dataset to a TimeSeriesTensor."""
        y = load_airline()
        times = y.index.to_timestamp().to_numpy()
        values = y.to_numpy().astype(np.float64)

        # Split into two halves and put them in a tensor
        mid = len(y) // 2
        ts1 = mpcf.TimeSeries(times[:mid], values[:mid])
        ts2 = mpcf.TimeSeries(times[mid:], values[mid:])
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])

        assert tensor.shape == (2,)

    def test_sktime_to_embedding(self):
        """Full pipeline: sktime dataset -> TimeSeries -> time delay embedding."""
        y = load_airline()
        times = y.index.to_timestamp().to_numpy()
        ts = mpcf.TimeSeries(times, y.to_numpy().astype(np.float64))

        cloud = mpcf.embed_time_delay(ts, dimension=3, delay=60 * 86400.0)
        pts = np.asarray(cloud[0])

        assert pts.ndim == 2
        assert pts.shape[1] == 3
