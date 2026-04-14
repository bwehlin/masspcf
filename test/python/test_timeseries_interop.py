"""Tests for TimeSeries interoperability with pandas, polars, xarray, and sktime."""

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
        t = np.datetime64("2024-01-02")
        assert tensor[0](t) == pytest.approx(23.0)
        assert tensor[1](t) == pytest.approx(1012.0)

    def test_dataset_to_multichannel(self):
        """Multiple xarray variables on the same time axis as channels."""
        ds = xr.Dataset(
            {
                "temp": (["time"], np.array([22.0, 23.0, 24.0])),
                "pressure": (["time"], np.array([1013.0, 1012.0, 1011.0])),
            },
            coords={"time": pd.date_range("2024-01-01", periods=3, freq="D")},
        )

        values = np.column_stack([ds[var].values for var in ds.data_vars])
        ts = mpcf.TimeSeries(ds.coords["time"].values, values)

        assert ts.n_channels == 2
        assert ts.n_times == 3
        t = np.datetime64("2024-01-02")
        np.testing.assert_array_almost_equal(ts(t), [23.0, 1012.0])


# ---------------------------------------------------------------------------
# polars (mirrors docs/ts_interop.rst example)
# ---------------------------------------------------------------------------


class TestPolars:
    def test_datetime_dataframe(self):
        """Polars DataFrame with Datetime column, as shown in docs."""
        df = pl.DataFrame({
            "timestamp": pl.datetime_range(
                datetime(2024, 1, 1), datetime(2024, 1, 2),
                interval="1h", eager=True,
            ),
            "value": np.arange(25, dtype=np.float64),
        })

        ts = mpcf.TimeSeries(
            df["timestamp"].to_numpy().astype("datetime64[us]"),
            df["value"].to_numpy(),
        )

        assert ts.n_times == 25
        assert ts(np.datetime64("2024-01-01T01:00:00")) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# sktime
# ---------------------------------------------------------------------------


class TestSktime:
    def test_period_index_conversion(self):
        """PeriodIndex -> datetime64 conversion, as shown in docs."""
        y = load_airline()  # pd.Series with PeriodIndex

        times = y.index.to_timestamp().to_numpy()
        ts = mpcf.TimeSeries(times, y.to_numpy().astype(float))

        assert ts.n_times == len(y)
        assert ts(times[0]) == pytest.approx(float(y.iloc[0]))
