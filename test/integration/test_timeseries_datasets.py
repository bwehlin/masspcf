"""Integration tests for TimeSeries with real-world datasets.

These tests download data from the internet on first run (cached locally
afterward).  They are intentionally kept out of ``test/python/`` so that
the fast unit-test suite can run without network access.

Run from the ``test/`` directory::

    python -m pytest integration/ -v
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.datasets import electrocardiogram
from sktime.datasets import load_airline, load_basic_motions

import masspcf as mpcf


# ---------------------------------------------------------------------------
# scipy ECG via pandas
# ---------------------------------------------------------------------------


class TestECGPandas:
    """Load the scipy ECG dataset through a pandas Series."""

    @pytest.fixture()
    def ecg_series(self):
        ecg = electrocardiogram()  # 108 000 samples, 360 Hz
        idx = pd.date_range(
            "2024-01-01", periods=len(ecg), freq=pd.Timedelta(1 / 360, "s"),
        )
        return pd.Series(ecg, index=idx)

    def test_construction(self, ecg_series):
        ts = mpcf.TimeSeries(
            ecg_series.index.to_numpy(), ecg_series.to_numpy(),
        )

        assert ts.n_times == 108_000

    def test_evaluation(self, ecg_series):
        ts = mpcf.TimeSeries(
            ecg_series.index.to_numpy(), ecg_series.to_numpy(),
        )

        # Evaluate at the first sample time
        t0 = np.datetime64(ecg_series.index[0], "ns")
        assert ts(t0) == pytest.approx(ecg_series.iloc[0])

    def test_embedding(self, ecg_series):
        # Use first 1000 samples for speed
        short = ecg_series.iloc[:1000]
        ts = mpcf.TimeSeries(
            short.index.to_numpy(), short.to_numpy(),
        )

        delay_samples = 10
        delay_s = delay_samples / 360.0
        cloud = mpcf.embed_time_delay(ts, dimension=3, delay=delay_s)
        pts = np.asarray(cloud[0])

        assert pts.ndim == 2
        assert pts.shape[1] == 3
        assert pts.shape[0] > 0


# ---------------------------------------------------------------------------
# xarray air temperature
# ---------------------------------------------------------------------------


class TestXarrayAirTemperature:
    """Use the NCEP reanalysis air-temperature dataset from xarray."""

    @pytest.fixture()
    def air_ds(self):
        return xr.tutorial.load_dataset("air_temperature")

    def test_single_grid_point(self, air_ds):
        """Extract one grid point as a TimeSeries."""
        da = air_ds["air"].sel(lat=40.0, lon=260.0)  # near Denver

        ts = mpcf.TimeSeries(
            da.coords["time"].values,
            da.values.astype(np.float64),
        )

        assert ts.n_times == 2920
        # First value should match the dataset
        assert ts(da.coords["time"].values[0]) == pytest.approx(
            float(da.values[0]))

    def test_multiple_grid_points_to_tensor(self, air_ds):
        """Multiple grid points become a TimeSeriesTensor."""
        lats = [40.0, 50.0, 60.0]
        series = []
        for lat in lats:
            da = air_ds["air"].sel(lat=lat, lon=260.0)
            series.append(mpcf.TimeSeries(
                da.coords["time"].values,
                da.values.astype(np.float64),
            ))

        tensor = mpcf.TimeSeriesTensor(series)
        assert tensor.shape == (3,)

    def test_multichannel_from_grid(self, air_ds):
        """Multiple grid points as channels of one multi-channel series."""
        lats = [40.0, 50.0]
        channels = np.vstack([
            air_ds["air"].sel(lat=lat, lon=260.0).values.astype(np.float64)[np.newaxis, :]
            for lat in lats
        ])
        times = air_ds.coords["time"].values

        ts = mpcf.TimeSeries(times, channels)

        assert ts.n_channels == 2
        assert ts.n_times == 2920


# ---------------------------------------------------------------------------
# sktime airline passengers
# ---------------------------------------------------------------------------


class TestSktimeAirline:
    """The classic Box-Jenkins airline passengers dataset."""

    @pytest.fixture()
    def airline(self):
        return load_airline()

    def test_construction(self, airline):
        times = airline.index.to_timestamp().to_numpy()
        ts = mpcf.TimeSeries(times, airline.to_numpy().astype(np.float64))

        assert ts.n_times == len(airline)

    def test_embedding_pipeline(self, airline):
        """Full pipeline: sktime -> TimeSeries -> embedding -> point cloud."""
        times = airline.index.to_timestamp().to_numpy()
        ts = mpcf.TimeSeries(times, airline.to_numpy().astype(np.float64))

        # 1-month delay in seconds (approximate)
        delay = 30 * 86400.0
        cloud = mpcf.embed_time_delay(ts, dimension=3, delay=delay)
        pts = np.asarray(cloud[0])

        assert pts.ndim == 2
        assert pts.shape[1] == 3
        assert pts.shape[0] > 0


# ---------------------------------------------------------------------------
# sktime BasicMotions (6-channel accelerometer + gyroscope)
# ---------------------------------------------------------------------------


class TestSktimeBasicMotions:
    """6-channel motion capture: 3-axis accelerometer + 3-axis gyroscope."""

    @pytest.fixture()
    def motions(self):
        X, y = load_basic_motions(return_type="numpy3d")
        # X shape: (80 instances, 6 channels, 100 timepoints)
        return X, y

    def test_multichannel_construction(self, motions):
        X, y = motions
        # X[i] shape is already (n_channels, n_timepoints) -- no transpose needed
        ts = mpcf.TimeSeries(X[0], start_time=0.0, time_step=0.1)

        assert ts.n_channels == 6
        assert ts.n_times == 100

    def test_multichannel_evaluation(self, motions):
        X, _ = motions
        ts = mpcf.TimeSeries(X[0], start_time=0.0, time_step=0.1)

        # Evaluate at a single time -- returns one value per channel
        result = ts(0.0)
        np.testing.assert_array_almost_equal(result, X[0][:, 0])

        # Evaluate at multiple times -- shape (n_channels, n_times)
        result = ts(np.array([0.0, 0.5, 1.0]))
        assert result.shape == (6, 3)

    def test_multichannel_embedding(self, motions):
        X, _ = motions
        ts = mpcf.TimeSeries(X[0], start_time=0.0, time_step=0.1)

        cloud = mpcf.embed_time_delay(ts, dimension=2, delay=0.3)
        pts = np.asarray(cloud[0])

        # Multi-channel: each point has dimension * n_channels coordinates
        assert pts.ndim == 2
        assert pts.shape[1] == 2 * 6  # dim=2, channels=6

    def test_instances_to_tensor(self, motions):
        """Each motion instance becomes one multi-channel TimeSeries in a tensor."""
        X, _ = motions
        series = [
            mpcf.TimeSeries(X[i], start_time=0.0, time_step=0.1)
            for i in range(X.shape[0])
        ]
        tensor = mpcf.TimeSeriesTensor(series)

        assert tensor.shape == (80,)
