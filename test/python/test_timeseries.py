import math

import numpy as np
import pytest

import masspcf as mpcf


class TestTimeSeriesConstruction:
    def test_from_values_default_params(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]))
        assert ts.n_times == 3
        assert ts.n_channels == 1
        assert ts.start_time == 0.0
        assert ts.dtype == mpcf.ts64

    def test_from_values_custom_start_and_step(self):
        ts = mpcf.TimeSeries(np.array([10.0, 20.0]),
                             start_time=5.0, time_step=0.5)
        assert ts.start_time == 5.0
        np.testing.assert_allclose(ts.times, [5.0, 5.5])

    def test_from_values_float32(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0], dtype=np.float32))
        assert ts.dtype == mpcf.ts32

    def test_from_values_explicit_dtype(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]), dtype=mpcf.ts32)
        assert ts.dtype == mpcf.ts32

    def test_from_times_and_values(self):
        times = np.array([5.0, 6.0, 8.0])
        values = np.array([100.0, 200.0, 300.0])
        ts = mpcf.TimeSeries(times, values)
        assert ts.start_time == 5.0
        assert ts.n_times == 3
        assert ts(5.0) == 100.0
        assert ts(6.5) == 200.0

    def test_from_times_and_values_datetime(self):
        times = np.array([
            "2024-01-01T00:00:00.000",
            "2024-01-01T00:00:00.010",
            "2024-01-01T00:00:00.020",
        ], dtype="datetime64[ms]")
        values = np.array([1.0, 2.0, 3.0])
        ts = mpcf.TimeSeries(times, values)
        assert ts.start_time == times[0]
        assert ts(times[1]) == 2.0

    def test_from_existing_timeseries(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0]),
                              start_time=3.0, time_step=0.1)
        ts2 = mpcf.TimeSeries(ts1)
        assert ts2.start_time == 3.0
        assert ts2 == ts1

    def test_from_list(self):
        ts = mpcf.TimeSeries([5.0, 10.0, 15.0])
        assert ts.n_times == 3
        assert ts.dtype == mpcf.ts64

    def test_from_values_datetime_start(self):
        epoch = np.datetime64("2024-01-01T00:00:00")
        step = np.timedelta64(10, "ms")
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]),
                             start_time=epoch, time_step=step)
        assert ts.start_time == epoch

    def test_invalid_time_step(self):
        with pytest.raises(Exception):
            mpcf.TimeSeries(np.array([1.0, 2.0]), time_step=-1.0)

    def test_invalid_data_type(self):
        with pytest.raises((TypeError, ValueError)):
            mpcf.TimeSeries(42)

    def test_times_values_length_mismatch(self):
        with pytest.raises(ValueError):
            mpcf.TimeSeries(np.array([1.0, 2.0]), np.array([1.0]))

    def test_times_values_needs_two_points(self):
        with pytest.raises(ValueError):
            mpcf.TimeSeries(np.array([1.0]), np.array([1.0]))

    def test_from_2d_values(self):
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        ts = mpcf.TimeSeries(values, start_time=0.0, time_step=1.0)
        assert ts.n_times == 3
        assert ts.n_channels == 2

    def test_from_times_and_2d_values(self):
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        ts = mpcf.TimeSeries(times, values)
        assert ts.n_times == 3
        assert ts.n_channels == 2


class TestTimeSeriesEvaluation:
    def test_eval_scalar_at_start(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]),
                             start_time=10.0, time_step=0.5)
        assert ts(10.0) == 1.0

    def test_eval_scalar_mid_interval(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]),
                             start_time=10.0, time_step=0.5)
        assert ts(10.25) == 1.0
        assert ts(10.5) == 2.0
        assert ts(10.75) == 2.0

    def test_eval_before_start_returns_nan(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]),
                             start_time=10.0, time_step=1.0)
        assert math.isnan(ts(9.0))

    def test_eval_after_end_returns_nan(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]),
                             start_time=10.0, time_step=1.0)
        assert math.isnan(ts(12.0))

    def test_eval_array(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]),
                             start_time=0.0, time_step=1.0)
        result = ts(np.array([0.0, 0.5, 1.0, 1.5]))
        np.testing.assert_array_equal(result, [1.0, 1.0, 2.0, 2.0])

    def test_eval_array_with_nan(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]),
                             start_time=5.0, time_step=1.0)
        result = ts(np.array([4.0, 5.0, 7.0]))
        assert math.isnan(result[0])
        assert result[1] == 1.0
        assert math.isnan(result[2])

    def test_eval_list(self):
        ts = mpcf.TimeSeries(np.array([10.0, 20.0]),
                             start_time=0.0, time_step=1.0)
        result = ts([0.0, 0.5])
        assert result[0] == 10.0

    def test_eval_from_times_values(self):
        times = np.array([5.0, 6.0, 8.0])
        values = np.array([100.0, 200.0, 300.0])
        ts = mpcf.TimeSeries(times, values)
        assert ts(5.0) == 100.0
        assert ts(6.5) == 200.0


class TestTimeSeriesMultiChannel:
    def test_eval_scalar_multi_channel(self):
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        ts = mpcf.TimeSeries(times, values)
        result = ts(0.5)
        np.testing.assert_array_equal(result, [1.0, 10.0])

    def test_eval_array_multi_channel(self):
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        ts = mpcf.TimeSeries(times, values)
        result = ts(np.array([0.5, 1.5]))
        # Shape: (n_channels, n_times) = (2, 2)
        assert result.shape == (2, 2)
        # Channel 0 at times [0.5, 1.5] -> [1.0, 2.0]
        np.testing.assert_array_equal(result[0], [1.0, 2.0])
        # Channel 1 at times [0.5, 1.5] -> [10.0, 20.0]
        np.testing.assert_array_equal(result[1], [10.0, 20.0])

    def test_eval_multi_channel_nan(self):
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        ts = mpcf.TimeSeries(times, values)
        result = ts(-1.0)
        assert all(math.isnan(v) for v in result)

    def test_from_regular_2d(self):
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        ts = mpcf.TimeSeries(values, start_time=5.0, time_step=0.5)
        assert ts.n_channels == 2
        assert ts.n_times == 3
        result = ts(5.0)
        np.testing.assert_array_equal(result, [1.0, 10.0])

    def test_repr_multi_channel(self):
        values = np.array([[1.0, 10.0], [2.0, 20.0]])
        ts = mpcf.TimeSeries(values, start_time=0.0, time_step=1.0)
        r = repr(ts)
        assert "n_channels=2" in r

    def test_values_property_2d(self):
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        ts = mpcf.TimeSeries(times, values)
        result = ts.values
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, values)

    def test_three_channels(self):
        times = np.array([0.0, 1.0])
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ts = mpcf.TimeSeries(times, values)
        assert ts.n_channels == 3
        result = ts(0.5)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_datetime_multi_channel(self):
        epoch = np.datetime64("2024-01-01T00:00:00")
        step = np.timedelta64(1, "s")
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        ts = mpcf.TimeSeries(values, start_time=epoch, time_step=step)
        assert ts.n_channels == 2
        t = np.datetime64("2024-01-01T00:00:01.500")
        result = ts(t)
        np.testing.assert_array_equal(result, [2.0, 20.0])

    def test_tensor_of_multi_channel(self):
        ts1 = mpcf.TimeSeries(
            np.array([0.0, 1.0]),
            np.array([[1.0, 10.0], [2.0, 20.0]]))
        ts2 = mpcf.TimeSeries(
            np.array([0.0, 1.0]),
            np.array([[3.0, 30.0], [4.0, 40.0]]))
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])
        result = tensor(0.5)
        # Shape: (2, 2) -- 2 series, 2 channels each
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[1.0, 10.0], [3.0, 30.0]])

    def test_tensor_of_multi_channel_array_eval(self):
        ts1 = mpcf.TimeSeries(
            np.array([0.0, 1.0, 2.0]),
            np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
        ts2 = mpcf.TimeSeries(
            np.array([0.0, 1.0, 2.0]),
            np.array([[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]]))
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])
        result = tensor(np.array([0.5, 1.5]))
        # Shape: (2, 2, 2) -- 2 series, 2 channels, 2 times
        assert result.shape == (2, 2, 2)
        # series 0, channel 0 -> [1.0, 2.0], channel 1 -> [10.0, 20.0]
        np.testing.assert_array_equal(result[0, 0], [1.0, 2.0])
        np.testing.assert_array_equal(result[0, 1], [10.0, 20.0])
        # series 1, channel 0 -> [4.0, 5.0], channel 1 -> [40.0, 50.0]
        np.testing.assert_array_equal(result[1, 0], [4.0, 5.0])
        np.testing.assert_array_equal(result[1, 1], [40.0, 50.0])


class TestTimeSeriesDatetime:
    def test_eval_datetime_precise(self):
        epoch = np.datetime64("2024-01-01T00:00:00")
        step = np.timedelta64(10, "ms")
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0, 4.0]),
                             start_time=epoch, time_step=step)

        t1 = np.datetime64("2024-01-01T00:00:00.010")
        assert ts(t1) == 2.0

        t2 = np.datetime64("2024-01-01T00:00:00.025")
        assert ts(t2) == 3.0

    def test_eval_datetime_before_start_nan(self):
        epoch = np.datetime64("2024-01-01T00:00:00")
        step = np.timedelta64(1, "s")
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]),
                             start_time=epoch, time_step=step)
        assert math.isnan(ts(np.datetime64("2023-12-31T23:59:59")))

    def test_eval_datetime_array(self):
        epoch = np.datetime64("2024-01-01T00:00:00")
        step = np.timedelta64(10, "ms")
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]),
                             start_time=epoch, time_step=step)

        times = np.array([
            "2024-01-01T00:00:00.000",
            "2024-01-01T00:00:00.010",
            "2024-01-01T00:00:00.020",
        ], dtype="datetime64[ms]")
        result = ts(times)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_from_times_values_datetime(self):
        times = np.array([
            "2024-06-15T08:00:00.000",
            "2024-06-15T08:00:00.500",
            "2024-06-15T08:00:01.000",
        ], dtype="datetime64[ms]")
        values = np.array([10.0, 20.0, 30.0])
        ts = mpcf.TimeSeries(times, values)
        assert ts(np.datetime64("2024-06-15T08:00:00.500")) == 20.0

    def test_times_property_datetime(self):
        epoch = np.datetime64("2024-01-01T00:00:00")
        step = np.timedelta64(10, "ms")
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]),
                             start_time=epoch, time_step=step)
        expected = np.array([
            "2024-01-01T00:00:00.000",
            "2024-01-01T00:00:00.010",
            "2024-01-01T00:00:00.020",
        ], dtype="datetime64[ms]")
        np.testing.assert_array_equal(ts.times, expected)

    def test_tensor_eval_datetime(self):
        epoch1 = np.datetime64("2024-06-15T08:00:00")
        epoch2 = np.datetime64("2024-06-15T08:00:02")
        step = np.timedelta64(500, "ms")

        ts1 = mpcf.TimeSeries(np.array([22.1, 22.3, 23.0, 22.8]),
                               start_time=epoch1, time_step=step)
        ts2 = mpcf.TimeSeries(np.array([21.0, 21.8, 22.5]),
                               start_time=epoch2, time_step=step)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])

        t = np.datetime64("2024-06-15T08:00:03")
        result = tensor(t)
        assert math.isnan(result[0])
        assert result[1] == 22.5

    def test_tensor_eval_datetime_array(self):
        epoch = np.datetime64("2024-01-01T00:00:00")
        step = np.timedelta64(1, "s")

        ts1 = mpcf.TimeSeries(np.array([10.0, 20.0, 30.0]),
                               start_time=epoch, time_step=step)
        ts2 = mpcf.TimeSeries(np.array([40.0, 50.0, 60.0]),
                               start_time=epoch, time_step=step)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])

        times = np.array([
            "2024-01-01T00:00:00.500",
            "2024-01-01T00:00:01.500",
        ], dtype="datetime64[ms]")
        result = tensor(times)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[10.0, 20.0], [40.0, 50.0]])

    def test_end_time_datetime(self):
        epoch = np.datetime64("2024-01-01T00:00:00")
        step = np.timedelta64(10, "ms")
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]),
                             start_time=epoch, time_step=step)
        # end_time is the C++ float end_time (in seconds)
        # For datetime series this is start_seconds + last_breakpoint * 1
        # We can just check it's reasonable
        assert ts.end_time > 0


class TestTimeSeriesProperties:
    def test_values_1d(self):
        ts = mpcf.TimeSeries(np.array([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(ts.values, [10.0, 20.0, 30.0])

    def test_times_numeric(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]),
                             start_time=5.0, time_step=0.5)
        np.testing.assert_allclose(ts.times, [5.0, 5.5, 6.0])

    def test_len(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]))
        assert len(ts) == 3

    def test_repr(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]))
        r = repr(ts)
        assert "TimeSeries" in r
        assert "start_time" in r

    def test_equality(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0]),
                              start_time=1.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(np.array([1.0, 2.0]),
                              start_time=1.0, time_step=1.0)
        assert ts1 == ts2

    def test_inequality_different_values(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0]))
        ts2 = mpcf.TimeSeries(np.array([1.0, 3.0]))
        assert ts1 != ts2

    def test_inequality_different_start(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0]), start_time=0.0)
        ts2 = mpcf.TimeSeries(np.array([1.0, 2.0]), start_time=1.0)
        assert ts1 != ts2


class TestTimeSeriesTensor:
    def test_construction_from_list(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0]),
                              start_time=0.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(np.array([3.0, 4.0]),
                              start_time=0.0, time_step=1.0)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])
        assert tensor.shape == (2,)

    def test_construction_from_nested_list(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]))
        tensor = mpcf.TimeSeriesTensor([[ts, ts], [ts, ts]])
        assert tensor.shape == (2, 2)

    def test_eval_scalar(self):
        ts1 = mpcf.TimeSeries(np.array([10.0, 20.0]),
                              start_time=0.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(np.array([30.0, 40.0]),
                              start_time=0.0, time_step=1.0)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])
        result = tensor(0.5)
        np.testing.assert_array_equal(result, [10.0, 30.0])

    def test_eval_with_different_starts(self):
        ts1 = mpcf.TimeSeries(np.array([10.0, 20.0]),
                              start_time=0.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(np.array([30.0, 40.0]),
                              start_time=5.0, time_step=1.0)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])

        result = tensor(0.5)
        assert result[0] == 10.0
        assert math.isnan(result[1])

    def test_eval_with_different_time_steps(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]),
                              start_time=0.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(np.array([10.0, 20.0, 30.0]),
                              start_time=0.0, time_step=0.5)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])

        result = tensor(0.5)
        assert result[0] == 1.0
        assert result[1] == 20.0

    def test_slicing_scalar(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0]),
                              start_time=3.0, time_step=0.5)
        ts2 = mpcf.TimeSeries(np.array([3.0, 4.0]),
                              start_time=5.0, time_step=1.0)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])

        elem = tensor[0]
        assert isinstance(elem, mpcf.TimeSeries)
        assert elem(3.0) == 1.0

    def test_slicing_range(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0]))
        ts2 = mpcf.TimeSeries(np.array([2.0, 3.0]))
        ts3 = mpcf.TimeSeries(np.array([3.0, 4.0]))
        tensor = mpcf.TimeSeriesTensor([ts1, ts2, ts3])

        sub = tensor[0:2]
        assert isinstance(sub, mpcf.TimeSeriesTensor)
        assert sub.shape == (2,)

    def test_start_times(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0]),
                              start_time=10.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(np.array([1.0, 2.0]),
                              start_time=20.0, time_step=1.0)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])
        np.testing.assert_array_equal(tensor.start_times, [10.0, 20.0])

    def test_end_times(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0]),
                              start_time=0.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]),
                              start_time=0.0, time_step=1.0)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])
        np.testing.assert_array_equal(tensor.end_times, [1.0, 2.0])

    def test_dtype(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]))
        tensor = mpcf.TimeSeriesTensor([ts])
        assert tensor.dtype == mpcf.ts64

    def test_eval_array(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]),
                              start_time=0.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(np.array([10.0, 20.0, 30.0]),
                              start_time=0.0, time_step=1.0)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])
        result = tensor(np.array([0.5, 1.5]))
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[1.0, 2.0], [10.0, 20.0]])


class TestTimeSeriesDtype:
    def test_ts32_dtype(self):
        assert str(mpcf.ts32) == "ts32"

    def test_ts64_dtype(self):
        assert str(mpcf.ts64) == "ts64"

    def test_dtype_inference_float32(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0], dtype=np.float32))
        assert ts.dtype == mpcf.ts32

    def test_dtype_inference_float64(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0], dtype=np.float64))
        assert ts.dtype == mpcf.ts64


class TestInterpolation:
    """Tests for TimeSeries interpolation modes."""

    def test_default_is_nearest(self):
        ts = mpcf.TimeSeries(np.array([0.0, 10.0, 20.0]))
        assert ts.interpolation == 'nearest'

    def test_nearest_step_function(self):
        ts = mpcf.TimeSeries(np.array([0.0, 10.0, 20.0]))
        assert ts(0.5) == 0.0
        assert ts(1.5) == 10.0

    def test_linear_basic(self):
        ts = mpcf.TimeSeries(np.array([0.0, 10.0, 20.0]),
                             interpolation='linear')
        assert ts(0.5) == pytest.approx(5.0)
        assert ts(1.5) == pytest.approx(15.0)

    def test_linear_at_breakpoints(self):
        ts = mpcf.TimeSeries(np.array([0.0, 10.0, 20.0]),
                             interpolation='linear')
        assert ts(0.0) == pytest.approx(0.0)
        assert ts(1.0) == pytest.approx(10.0)
        assert ts(2.0) == pytest.approx(20.0)

    def test_linear_out_of_bounds_nan(self):
        ts = mpcf.TimeSeries(
            np.array([0.0, 10.0, 20.0]),
            start_time=1.0, time_step=1.0,
            interpolation='linear')
        assert np.isnan(ts(0.5))
        assert np.isnan(ts(4.0))

    def test_linear_multi_channel(self):
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([
            [0.0, 100.0],
            [10.0, 200.0],
            [20.0, 300.0],
        ])
        ts = mpcf.TimeSeries(times, values, interpolation='linear')
        result = ts(0.5)
        np.testing.assert_allclose(result, [5.0, 150.0])

    def test_linear_last_breakpoint(self):
        ts = mpcf.TimeSeries(np.array([0.0, 10.0, 20.0]),
                             interpolation='linear')
        assert ts(2.0) == pytest.approx(20.0)

    def test_settable(self):
        ts = mpcf.TimeSeries(np.array([0.0, 10.0, 20.0]))
        assert ts.interpolation == 'nearest'
        assert ts(0.5) == 0.0

        ts.interpolation = 'linear'
        assert ts.interpolation == 'linear'
        assert ts(0.5) == pytest.approx(5.0)

    def test_constructor_param(self):
        ts = mpcf.TimeSeries(np.array([0.0, 10.0, 20.0]),
                             interpolation='linear')
        assert ts.interpolation == 'linear'

    def test_equality_different_modes(self):
        ts1 = mpcf.TimeSeries(np.array([0.0, 10.0]))
        ts2 = mpcf.TimeSeries(np.array([0.0, 10.0]),
                              interpolation='linear')
        assert ts1 != ts2

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            mpcf.TimeSeries(np.array([0.0, 10.0]),
                            interpolation='cubic')

    def test_linear_array_eval(self):
        ts = mpcf.TimeSeries(np.array([0.0, 10.0, 20.0]),
                             interpolation='linear')
        result = ts(np.array([0.5, 1.0, 1.5]))
        np.testing.assert_allclose(result, [5.0, 10.0, 15.0])

    def test_repr_shows_linear(self):
        ts = mpcf.TimeSeries(np.array([0.0, 10.0]),
                             interpolation='linear')
        assert "interpolation='linear'" in repr(ts)

    def test_repr_hides_nearest(self):
        ts = mpcf.TimeSeries(np.array([0.0, 10.0]))
        assert 'interpolation' not in repr(ts)


class TestEmbedTimeDelay:
    """Tests for mpcf.embed_time_delay (Takens time delay embedding).

    Embedding vector at time t (backward-looking):
        [x(t-(d-1)*tau), ..., x(t-tau), x(t)]
    Valid times: t >= start + (d-1)*tau.
    """

    # --- Univariate, no windowing ---

    def test_univariate_basic(self):
        # values [1,2,3,4,5] at times 0..4, d=2, delay=1.0
        # Valid t: [1, 4]. Base times: 1, 2, 3, 4.
        # Vectors: [x(0),x(1)], [x(1),x(2)], [x(2),x(3)], [x(3),x(4)]
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = mpcf.embed_time_delay(ts, dimension=2, delay=1.0)
        assert isinstance(result, mpcf.PointCloudTensor)
        assert result.shape == (1,)
        cloud = np.asarray(result[0])
        expected = np.array([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ])
        np.testing.assert_allclose(cloud, expected)

    def test_univariate_delay_2(self):
        # values [1,2,3,4,5] at times 0..4, d=2, delay=2.0
        # Valid t: [2, 4]. Base times: 2, 3, 4.
        # Vectors: [x(0),x(2)], [x(1),x(3)], [x(2),x(4)]
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = mpcf.embed_time_delay(ts, dimension=2, delay=2.0)
        cloud = np.asarray(result[0])
        expected = np.array([
            [1.0, 3.0],
            [2.0, 4.0],
            [3.0, 5.0],
        ])
        np.testing.assert_allclose(cloud, expected)

    def test_higher_dimension(self):
        # values [1,2,3,4,5] at times 0..4, d=3, delay=1.0
        # Valid t: [2, 4]. Base times: 2, 3, 4.
        # Vectors: [x(0),x(1),x(2)], [x(1),x(2),x(3)], [x(2),x(3),x(4)]
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = mpcf.embed_time_delay(ts, dimension=3, delay=1.0)
        cloud = np.asarray(result[0])
        expected = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ])
        np.testing.assert_allclose(cloud, expected)

    def test_dimension_1(self):
        # d=1, no lookback. Every time point is valid.
        # Each embedding vector is just [x(t)].
        ts = mpcf.TimeSeries(np.array([10.0, 20.0, 30.0]))
        result = mpcf.embed_time_delay(ts, dimension=1, delay=1.0)
        cloud = np.asarray(result[0])
        expected = np.array([[10.0], [20.0], [30.0]])
        np.testing.assert_allclose(cloud, expected)

    # --- Multivariate ---

    def test_multivariate(self):
        # 5-point, 2-channel series, d=2, delay=1.0
        # channels: ch0=[1,2,3,4,5], ch1=[10,20,30,40,50]
        # Valid t: [1, 4]. Base times: 1, 2, 3, 4.
        # At t=1: [x(0,:), x(1,:)] = [1, 10, 2, 20]
        # At t=2: [x(1,:), x(2,:)] = [2, 20, 3, 30]
        # etc.
        times = np.arange(5, dtype=np.float64)
        values = np.column_stack([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ])
        ts = mpcf.TimeSeries(times, values)
        result = mpcf.embed_time_delay(ts, dimension=2, delay=1.0)
        cloud = np.asarray(result[0])
        expected = np.array([
            [1.0, 10.0, 2.0, 20.0],
            [2.0, 20.0, 3.0, 30.0],
            [3.0, 30.0, 4.0, 40.0],
            [4.0, 40.0, 5.0, 50.0],
        ])
        np.testing.assert_allclose(cloud, expected)

    # --- Non-unit time_step ---

    def test_non_unit_time_step(self):
        # time_step=0.5, times=[0, 0.5, 1.0, 1.5, 2.0], delay=1.0
        # delay=1.0 spans 2 time steps. d=2.
        # Valid t: [1.0, 2.0]. Base times at: 1.0, 1.5, 2.0.
        # At t=1.0: [x(0.0), x(1.0)] = [1, 3]
        # At t=1.5: [x(0.5), x(1.5)] = [2, 4]
        # At t=2.0: [x(1.0), x(2.0)] = [3, 5]
        ts = mpcf.TimeSeries(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            start_time=0.0, time_step=0.5)
        result = mpcf.embed_time_delay(ts, dimension=2, delay=1.0)
        cloud = np.asarray(result[0])
        expected = np.array([
            [1.0, 3.0],
            [2.0, 4.0],
            [3.0, 5.0],
        ])
        np.testing.assert_allclose(cloud, expected)

    # --- Windowing (backward from end) ---

    def test_windowing_basic(self):
        # 10 points at times 0..9, d=2, delay=1.0, window=3.0
        # Valid t: [1, 9]. Valid range length = 8.
        # stride defaults to window=3.0.
        # Windows backward from end:
        #   win 2 (last):  [9-3, 9] = [6, 9] -> base times 6,7,8,9
        #   win 1:         [3, 6)           -> base times 3,4,5
        #   win 0 (first): [1, 3)           -> base times 1,2 (partial)
        vals = np.arange(1.0, 11.0)
        ts = mpcf.TimeSeries(vals)
        result = mpcf.embed_time_delay(
            ts, dimension=2, delay=1.0, window=3.0)
        assert result.shape == (3,)

        # Last window: [6, 9], base times 6,7,8,9
        cloud2 = np.asarray(result[2])
        assert cloud2.shape[1] == 2
        # t=6: [x(5),x(6)]=[6,7]; t=7: [7,8]; t=8: [8,9]; t=9: [9,10]
        expected2 = np.array([
            [6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0]])
        np.testing.assert_allclose(cloud2, expected2)

    def test_windowing_with_stride(self):
        # 10 points at times 0..9, d=2, delay=1.0
        # window=4.0, stride=2.0 (overlapping)
        # Valid t: [1, 9]. Valid range = [1, 9].
        # Windows backward from 9:
        #   last:  [5, 9] -> base 5,6,7,8,9
        #   prev:  [3, 7) -> base 3,4,5,6
        #   first: [1, 5) -> base 1,2,3,4
        vals = np.arange(1.0, 11.0)
        ts = mpcf.TimeSeries(vals)
        result = mpcf.embed_time_delay(
            ts, dimension=2, delay=1.0, window=4.0, stride=2.0)
        assert result.shape == (3,)

    # --- Validation ---

    def test_dimension_zero_raises(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError):
            mpcf.embed_time_delay(ts, dimension=0, delay=1.0)

    def test_delay_zero_raises(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError):
            mpcf.embed_time_delay(ts, dimension=2, delay=0.0)

    def test_delay_negative_raises(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError):
            mpcf.embed_time_delay(ts, dimension=2, delay=-1.0)

    def test_insufficient_time_range_raises(self):
        # 2 points, d=3, delay=1.0 needs valid t >= 2 but end=1
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]))
        with pytest.raises(ValueError):
            mpcf.embed_time_delay(ts, dimension=3, delay=1.0)

    # --- TimeSeriesTensor ---

    def test_tensor_common_domain(self):
        # ts1: times [0..4], ts2: times [2..6]
        # Common domain: [2, 4]. With d=2, delay=1.0:
        # Valid t: [3, 4] within common domain.
        ts1 = mpcf.TimeSeries(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            start_time=0.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            start_time=2.0, time_step=1.0)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])
        result = mpcf.embed_time_delay(tensor, dimension=2, delay=1.0)
        assert isinstance(result, mpcf.PointCloudTensor)
        assert result.shape == (2,)

        # ts1 base times in [3, 4]: t=3, t=4
        cloud0 = np.asarray(result[0])
        expected0 = np.array([
            [3.0, 4.0],  # t=3: [x(2), x(3)]
            [4.0, 5.0],  # t=4: [x(3), x(4)]
        ])
        np.testing.assert_allclose(cloud0, expected0)

        # ts2 base times in [3, 4]: t=3, t=4
        # ts2 values: time 2->10, 3->20, 4->30
        cloud1 = np.asarray(result[1])
        expected1 = np.array([
            [10.0, 20.0],  # t=3: [ts2(2), ts2(3)] = [10, 20]
            [20.0, 30.0],  # t=4: [ts2(3), ts2(4)] = [20, 30]
        ])
        np.testing.assert_allclose(cloud1, expected1)

    def test_tensor_different_sampling_rates(self):
        # ts1: time_step=1.0, times [0,1,2,3,4]
        # ts2: time_step=0.5, times [0,0.5,1,1.5,2,2.5,3,3.5,4]
        # Common domain: [0, 4]. d=2, delay=1.0.
        # Valid t: [1, 4].
        # ts1 has base times 1,2,3,4 -> 4 points
        # ts2 has base times 1,1.5,2,2.5,3,3.5,4 -> 7 points
        ts1 = mpcf.TimeSeries(
            np.arange(5, dtype=np.float64),
            start_time=0.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(
            np.arange(9, dtype=np.float64),
            start_time=0.0, time_step=0.5)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])
        result = mpcf.embed_time_delay(tensor, dimension=2, delay=1.0)
        assert result.shape == (2,)

        cloud0 = np.asarray(result[0])
        assert cloud0.shape == (4, 2)

        cloud1 = np.asarray(result[1])
        assert cloud1.shape == (7, 2)

    def test_tensor_with_windowing(self):
        # 2 series, windowed -> output shape (2, n_windows)
        ts1 = mpcf.TimeSeries(np.arange(10, dtype=np.float64))
        ts2 = mpcf.TimeSeries(np.arange(10, dtype=np.float64) * 10)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])
        result = mpcf.embed_time_delay(
            tensor, dimension=2, delay=1.0, window=3.0)
        assert result.shape[0] == 2
        assert result.shape[1] >= 2

    # --- dtype ---

    def test_ts32_produces_pcloud32(self):
        ts = mpcf.TimeSeries(
            np.array([1.0, 2.0, 3.0], dtype=np.float32))
        result = mpcf.embed_time_delay(ts, dimension=2, delay=1.0)
        assert result.dtype == mpcf.pcloud32

    def test_ts64_produces_pcloud64(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]))
        result = mpcf.embed_time_delay(ts, dimension=2, delay=1.0)
        assert result.dtype == mpcf.pcloud64

    # --- Datetime ---

    def test_datetime_delay(self):
        times = np.array([
            "2024-01-01T00:00:00",
            "2024-01-01T00:00:01",
            "2024-01-01T00:00:02",
            "2024-01-01T00:00:03",
            "2024-01-01T00:00:04",
        ], dtype="datetime64[s]")
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ts = mpcf.TimeSeries(times, values)
        result = mpcf.embed_time_delay(
            ts, dimension=2, delay=np.timedelta64(1, 's'))
        cloud = np.asarray(result[0])
        expected = np.array([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ])
        np.testing.assert_allclose(cloud, expected)
