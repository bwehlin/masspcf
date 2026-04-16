#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pickle

import numpy as np

import masspcf as mpcf
from masspcf.persistence import Barcode
from masspcf.timeseries import TimeSeries, TimeSeriesTensor


# --- Helpers ---


def _pickle_roundtrip(obj):
    data = pickle.dumps(obj)
    return pickle.loads(data)


def _assert_tensor_roundtrip(t):
    restored = _pickle_roundtrip(t)
    assert type(restored) is type(t)
    assert restored.dtype == t.dtype
    assert restored.shape == t.shape
    assert t.array_equal(restored)


# --- Float tensors ---


def test_pickle_float32_tensor():
    _assert_tensor_roundtrip(
        mpcf.FloatTensor(np.array([1.0, 2.0, 3.0], dtype=np.float32)))


def test_pickle_float64_tensor():
    _assert_tensor_roundtrip(
        mpcf.FloatTensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)))


def test_pickle_float_tensor_3d():
    _assert_tensor_roundtrip(
        mpcf.FloatTensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4)))


# --- Int tensors ---


def test_pickle_int32_tensor():
    _assert_tensor_roundtrip(
        mpcf.IntTensor(np.array([10, 20, 30], dtype=np.int32)))


def test_pickle_int64_tensor():
    _assert_tensor_roundtrip(
        mpcf.IntTensor(np.array([[1, 2], [3, 4]], dtype=np.int64)))


# --- Bool tensor ---


def test_pickle_bool_tensor():
    _assert_tensor_roundtrip(
        mpcf.BoolTensor(np.array([[True, False], [False, True]])))


# --- PCF tensors ---


def test_pickle_pcf32_tensor():
    f = mpcf.Pcf(np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32))
    g = mpcf.Pcf(np.array([[0.0, 3.0], [2.0, 4.0]], dtype=np.float32))
    _assert_tensor_roundtrip(mpcf.PcfTensor([f, g]))


def test_pickle_pcf64_tensor():
    f = mpcf.Pcf(np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64))
    _assert_tensor_roundtrip(mpcf.PcfTensor([f]))


def test_pickle_pcf_tensor_2d():
    fs = [mpcf.Pcf(np.array([[0, float(i)], [1, float(i + 1)]], dtype=np.float32))
          for i in range(6)]
    t = mpcf.PcfTensor(fs).reshape((2, 3))
    _assert_tensor_roundtrip(t)


# --- Standalone Pcf ---


def test_pickle_pcf_f32():
    f = mpcf.Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.5]], dtype=np.float32))
    restored = _pickle_roundtrip(f)
    assert isinstance(restored, mpcf.Pcf)
    assert f == restored


def test_pickle_pcf_f64():
    f = mpcf.Pcf(np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64))
    restored = _pickle_roundtrip(f)
    assert isinstance(restored, mpcf.Pcf)
    assert f == restored


def test_pickle_pcf_i32():
    f = mpcf.Pcf(np.array([[0, 1], [2, 3]], dtype=np.int32))
    restored = _pickle_roundtrip(f)
    assert isinstance(restored, mpcf.Pcf)
    assert f == restored


def test_pickle_pcf_i64():
    f = mpcf.Pcf(np.array([[0, 1], [2, 3]], dtype=np.int64))
    restored = _pickle_roundtrip(f)
    assert isinstance(restored, mpcf.Pcf)
    assert f == restored


# --- Barcode ---


def test_pickle_barcode_f64():
    bc = Barcode(np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float64))
    restored = _pickle_roundtrip(bc)
    assert isinstance(restored, Barcode)
    assert np.array_equal(np.asarray(bc), np.asarray(restored))


def test_pickle_barcode_f32():
    bc = Barcode(np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float32))
    restored = _pickle_roundtrip(bc)
    assert isinstance(restored, Barcode)
    assert np.array_equal(np.asarray(bc), np.asarray(restored))


# --- DistanceMatrix ---


def test_pickle_distance_matrix():
    dm = mpcf.DistanceMatrix(3, dtype=mpcf.float64)
    dm[0, 1] = 1.0
    dm[0, 2] = 2.0
    dm[1, 2] = 3.0
    restored = _pickle_roundtrip(dm)
    assert isinstance(restored, mpcf.DistanceMatrix)
    assert restored.size == dm.size
    assert restored[0, 1] == 1.0
    assert restored[0, 2] == 2.0
    assert restored[1, 2] == 3.0


# --- SymmetricMatrix ---


def test_pickle_symmetric_matrix():
    sm = mpcf.SymmetricMatrix(3, dtype=mpcf.float64)
    sm[0, 0] = 1.0
    sm[0, 1] = 2.0
    sm[1, 1] = 3.0
    restored = _pickle_roundtrip(sm)
    assert isinstance(restored, mpcf.SymmetricMatrix)
    assert restored.size == sm.size
    assert restored[0, 0] == 1.0
    assert restored[0, 1] == 2.0
    assert restored[1, 1] == 3.0


# --- TimeSeries ---


def test_pickle_timeseries_f32():
    ts = TimeSeries(np.array([1.0, 2.0, 3.0], dtype=np.float32),
                    start_time=0.0, time_step=1.0, dtype=mpcf.float32)
    restored = _pickle_roundtrip(ts)
    assert isinstance(restored, TimeSeries)
    assert restored == ts


def test_pickle_timeseries_f64():
    ts = TimeSeries(np.array([10.0, 20.0, 30.0], dtype=np.float64),
                    start_time=5.0, time_step=0.5, dtype=mpcf.float64)
    restored = _pickle_roundtrip(ts)
    assert isinstance(restored, TimeSeries)
    assert restored == ts


def test_pickle_timeseries_multichannel():
    vals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    ts = TimeSeries(vals, start_time=0.0, time_step=1.0)
    restored = _pickle_roundtrip(ts)
    assert isinstance(restored, TimeSeries)
    assert restored == ts
    assert restored.n_channels == 2
    np.testing.assert_array_equal(restored.values, vals)


def test_pickle_timeseries_linear_interpolation():
    ts = TimeSeries(np.array([1.0, 2.0, 3.0]),
                    start_time=0.0, time_step=1.0,
                    interpolation='linear')
    restored = _pickle_roundtrip(ts)
    assert isinstance(restored, TimeSeries)
    assert restored.interpolation == 'linear'
    assert restored == ts


# --- TimeSeriesTensor ---


def test_pickle_timeseries32_tensor():
    ts1 = TimeSeries(np.array([1.0, 2.0], dtype=np.float32),
                     start_time=0.0, time_step=1.0, dtype=mpcf.float32)
    ts2 = TimeSeries(np.array([3.0, 4.0], dtype=np.float32),
                     start_time=0.0, time_step=1.0, dtype=mpcf.float32)
    _assert_tensor_roundtrip(TimeSeriesTensor([ts1, ts2]))


def test_pickle_timeseries64_tensor():
    ts1 = TimeSeries(np.array([1.0, 2.0], dtype=np.float64),
                     start_time=0.0, time_step=1.0, dtype=mpcf.float64)
    ts2 = TimeSeries(np.array([3.0, 4.0], dtype=np.float64),
                     start_time=0.0, time_step=1.0, dtype=mpcf.float64)
    _assert_tensor_roundtrip(TimeSeriesTensor([ts1, ts2]))
