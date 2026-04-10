import math

import numpy as np
import pytest

import masspcf as mpcf


class TestTimeSeriesConstruction:
    def test_from_1d_values_default_params(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]))
        assert ts.size == 3
        assert ts.start_time == 0.0
        assert ts.time_step == 1.0
        assert ts.dtype == mpcf.ts64

    def test_from_1d_values_custom_epoch_and_step(self):
        ts = mpcf.TimeSeries(np.array([10.0, 20.0]), start_time=5.0, time_step=0.5)
        assert ts.start_time == 5.0
        assert ts.time_step == 0.5
        assert ts.start_time == 5.0
        assert ts.end_time == 5.5

    def test_from_1d_float32(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0], dtype=np.float32))
        assert ts.dtype == mpcf.ts32

    def test_from_1d_explicit_dtype(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]), dtype=mpcf.ts32)
        assert ts.dtype == mpcf.ts32

    def test_from_n2_pairs_numeric(self):
        data = np.array([[5.0, 100.0], [6.0, 200.0], [8.0, 300.0]])
        ts = mpcf.TimeSeries(data, time_step=1.0)
        assert ts.start_time == 5.0
        assert ts.start_time == 5.0
        assert ts.end_time == 8.0
        assert ts.size == 3

    def test_from_pcf(self):
        pcf = mpcf.Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.5]]))
        ts = mpcf.TimeSeries(pcf, start_time=10.0, time_step=2.0)
        assert ts.start_time == 10.0
        assert ts.time_step == 2.0
        assert ts.size == 3

    def test_from_existing_timeseries(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0]), start_time=3.0, time_step=0.1)
        ts2 = mpcf.TimeSeries(ts1)
        assert ts2.start_time == 3.0
        assert ts2.time_step == 0.1
        assert ts2 == ts1

    def test_from_list(self):
        ts = mpcf.TimeSeries([5.0, 10.0, 15.0])
        assert ts.size == 3
        assert ts.dtype == mpcf.ts64

    def test_from_n2_datetime64(self):
        epoch = np.datetime64("2024-01-01T00:00:00")
        step = np.timedelta64(10, "ms")
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]), start_time=epoch, time_step=step)
        assert ts.start_time == epoch
        assert ts.time_step == step
        assert ts.start_time == epoch

    def test_invalid_time_step(self):
        with pytest.raises(Exception):
            mpcf.TimeSeries(np.array([1.0]), time_step=0.0)

    def test_invalid_data_type(self):
        with pytest.raises(TypeError):
            mpcf.TimeSeries(42)


class TestTimeSeriesEvaluation:
    def test_eval_scalar_at_epoch(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]), start_time=10.0, time_step=0.5)
        assert ts(10.0) == 1.0

    def test_eval_scalar_mid_interval(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]), start_time=10.0, time_step=0.5)
        assert ts(10.25) == 1.0  # still in first interval
        assert ts(10.5) == 2.0   # second interval
        assert ts(10.75) == 2.0

    def test_eval_before_start_returns_nan(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]), start_time=10.0, time_step=1.0)
        assert math.isnan(ts(9.0))

    def test_eval_after_end_returns_nan(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]), start_time=10.0, time_step=1.0)
        # end_time = 10.0 + 1 * 1.0 = 11.0
        assert math.isnan(ts(12.0))

    def test_eval_array(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]), start_time=0.0, time_step=1.0)
        result = ts(np.array([0.0, 0.5, 1.0, 1.5]))
        np.testing.assert_array_equal(result, [1.0, 1.0, 2.0, 2.0])

    def test_eval_array_with_nan(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]), start_time=5.0, time_step=1.0)
        result = ts(np.array([4.0, 5.0, 7.0]))
        assert math.isnan(result[0])
        assert result[1] == 1.0
        assert math.isnan(result[2])

    def test_eval_list(self):
        ts = mpcf.TimeSeries(np.array([10.0, 20.0]), start_time=0.0, time_step=1.0)
        result = ts([0.0, 0.5])
        assert result[0] == 10.0

    def test_eval_from_n2_pairs(self):
        data = np.array([[5.0, 100.0], [6.0, 200.0], [8.0, 300.0]])
        ts = mpcf.TimeSeries(data, time_step=1.0)
        assert ts(5.0) == 100.0
        assert ts(6.5) == 200.0
        assert ts(8.0) == pytest.approx(300.0)  # at last breakpoint, still in domain

    def test_eval_from_pcf(self):
        pcf = mpcf.Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.5]]))
        ts = mpcf.TimeSeries(pcf, start_time=10.0, time_step=2.0)
        assert ts(10.0) == 1.0   # pcf_t = 0
        assert ts(12.0) == 2.0   # pcf_t = 1
        assert ts(15.0) == 2.0   # pcf_t = 2.5 -> in [1, 3) interval, value = 2.0
        assert ts(16.0) == 0.5   # pcf_t = 3.0 -> at breakpoint t=3, value = 0.5


class TestTimeSeriesDatetime:
    def test_eval_datetime_precise(self):
        epoch = np.datetime64("2024-01-01T00:00:00")
        step = np.timedelta64(10, "ms")
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0, 4.0]), start_time=epoch, time_step=step)

        t1 = np.datetime64("2024-01-01T00:00:00.010")  # +10ms -> pcf_t=1
        assert ts(t1) == 2.0

        t2 = np.datetime64("2024-01-01T00:00:00.025")  # +25ms -> pcf_t=2.5
        assert ts(t2) == 3.0

    def test_eval_datetime_before_epoch_nan(self):
        epoch = np.datetime64("2024-01-01T00:00:00")
        step = np.timedelta64(1, "s")
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]), start_time=epoch, time_step=step)
        assert math.isnan(ts(np.datetime64("2023-12-31T23:59:59")))

    def test_eval_datetime_array(self):
        epoch = np.datetime64("2024-01-01T00:00:00")
        step = np.timedelta64(10, "ms")
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]), start_time=epoch, time_step=step)

        times = np.array([
            "2024-01-01T00:00:00.000",
            "2024-01-01T00:00:00.010",
            "2024-01-01T00:00:00.020",
        ], dtype="datetime64[ms]")
        result = ts(times)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_times_property_datetime(self):
        epoch = np.datetime64("2024-01-01T00:00:00")
        step = np.timedelta64(10, "ms")
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]), start_time=epoch, time_step=step)
        expected = np.array([
            "2024-01-01T00:00:00.000",
            "2024-01-01T00:00:00.010",
            "2024-01-01T00:00:00.020",
        ], dtype="datetime64[ms]")
        np.testing.assert_array_equal(ts.times, expected)

    def test_end_time_datetime(self):
        epoch = np.datetime64("2024-01-01T00:00:00")
        step = np.timedelta64(10, "ms")
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]), start_time=epoch, time_step=step)
        assert ts.end_time == np.datetime64("2024-01-01T00:00:00.020")


class TestTimeSeriesProperties:
    def test_values(self):
        ts = mpcf.TimeSeries(np.array([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(ts.values, [10.0, 20.0, 30.0])

    def test_times_numeric(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]), start_time=5.0, time_step=0.5)
        np.testing.assert_allclose(ts.times, [5.0, 5.5, 6.0])

    def test_len(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]))
        assert len(ts) == 3

    def test_pcf_property(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]), start_time=5.0, time_step=1.0)
        pcf = ts.pcf
        assert isinstance(pcf, mpcf.Pcf)
        assert pcf.size == 2

    def test_repr(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]))
        r = repr(ts)
        assert "TimeSeries" in r
        assert "start_time" in r

    def test_equality(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0]), start_time=1.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(np.array([1.0, 2.0]), start_time=1.0, time_step=1.0)
        assert ts1 == ts2

    def test_inequality_different_values(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0]))
        ts2 = mpcf.TimeSeries(np.array([1.0, 3.0]))
        assert ts1 != ts2

    def test_inequality_different_epoch(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0]), start_time=0.0)
        ts2 = mpcf.TimeSeries(np.array([1.0, 2.0]), start_time=1.0)
        assert ts1 != ts2


class TestTimeSeriesTensor:
    def test_construction_from_list(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0]), start_time=0.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(np.array([3.0, 4.0]), start_time=0.0, time_step=1.0)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])
        assert tensor.shape == (2,)

    def test_construction_from_nested_list(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]))
        tensor = mpcf.TimeSeriesTensor([[ts, ts], [ts, ts]])
        assert tensor.shape == (2, 2)

    def test_eval_scalar(self):
        ts1 = mpcf.TimeSeries(np.array([10.0, 20.0]), start_time=0.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(np.array([30.0, 40.0]), start_time=0.0, time_step=1.0)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])
        result = tensor(0.5)
        np.testing.assert_array_equal(result, [10.0, 30.0])

    def test_eval_with_different_epochs(self):
        ts1 = mpcf.TimeSeries(np.array([10.0, 20.0]), start_time=0.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(np.array([30.0, 40.0]), start_time=5.0, time_step=1.0)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])

        # At t=0.5: ts1 is in range, ts2 is before its epoch -> NaN
        result = tensor(0.5)
        assert result[0] == 10.0
        assert math.isnan(result[1])

    def test_eval_with_different_time_steps(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]), start_time=0.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(np.array([10.0, 20.0, 30.0]), start_time=0.0, time_step=0.5)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])

        result = tensor(0.5)
        assert result[0] == 1.0   # pcf_t = 0.5 -> first interval
        assert result[1] == 20.0  # pcf_t = 1.0 -> second interval

    def test_slicing_scalar(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0]), start_time=3.0, time_step=0.5)
        ts2 = mpcf.TimeSeries(np.array([3.0, 4.0]), start_time=5.0, time_step=1.0)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])

        elem = tensor[0]
        assert isinstance(elem, mpcf.TimeSeries)
        assert elem(3.0) == 1.0

    def test_slicing_range(self):
        ts1 = mpcf.TimeSeries(np.array([1.0]))
        ts2 = mpcf.TimeSeries(np.array([2.0]))
        ts3 = mpcf.TimeSeries(np.array([3.0]))
        tensor = mpcf.TimeSeriesTensor([ts1, ts2, ts3])

        sub = tensor[0:2]
        assert isinstance(sub, mpcf.TimeSeriesTensor)
        assert sub.shape == (2,)

    def test_start_times(self):
        ts1 = mpcf.TimeSeries(np.array([1.0]), start_time=10.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(np.array([1.0]), start_time=20.0, time_step=1.0)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])
        np.testing.assert_array_equal(tensor.start_times, [10.0, 20.0])

    def test_end_times(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0]), start_time=0.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]), start_time=0.0, time_step=1.0)
        tensor = mpcf.TimeSeriesTensor([ts1, ts2])
        np.testing.assert_array_equal(tensor.end_times, [1.0, 2.0])

    def test_dtype(self):
        ts = mpcf.TimeSeries(np.array([1.0, 2.0]))
        tensor = mpcf.TimeSeriesTensor([ts])
        assert tensor.dtype == mpcf.ts64

    def test_eval_array(self):
        ts1 = mpcf.TimeSeries(np.array([1.0, 2.0, 3.0]), start_time=0.0, time_step=1.0)
        ts2 = mpcf.TimeSeries(np.array([10.0, 20.0, 30.0]), start_time=0.0, time_step=1.0)
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
        ts = mpcf.TimeSeries(np.array([1.0], dtype=np.float32))
        assert ts.dtype == mpcf.ts32

    def test_dtype_inference_float64(self):
        ts = mpcf.TimeSeries(np.array([1.0], dtype=np.float64))
        assert ts.dtype == mpcf.ts64
