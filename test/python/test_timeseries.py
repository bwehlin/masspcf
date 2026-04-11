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
