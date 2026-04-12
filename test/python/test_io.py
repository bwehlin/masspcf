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

import io

import numpy as np
import pytest

import masspcf as mpcf
from masspcf.timeseries import TimeSeries, TimeSeriesTensor


def test_float32_tensor_roundtrip():
    original = mpcf.FloatTensor(np.random.randn(2, 3))

    buf = io.BytesIO()
    mpcf.save(original, buf)

    buf.seek(0)  # rewind before reading!
    restored = mpcf.load(buf)

    assert original.array_equal(restored)


def _make_symmetric_matrix(n, dtype):
    mat = mpcf.SymmetricMatrix(n, dtype=dtype)
    for i in range(n):
        for j in range(i + 1):
            mat[i, j] = float(i * n + j)
    return mat


@pytest.mark.parametrize("symmat_dtype, scalar_dtype", [
    (mpcf.symmat32, mpcf.float32),
    (mpcf.symmat64, mpcf.float64),
])
def test_symmetric_matrix_tensor_roundtrip(symmat_dtype, scalar_dtype):
    T = mpcf.zeros((2,), dtype=symmat_dtype)
    T[0] = _make_symmetric_matrix(3, scalar_dtype)
    T[1] = _make_symmetric_matrix(4, scalar_dtype)

    buf = io.BytesIO()
    mpcf.save(T, buf)

    buf.seek(0)
    restored = mpcf.load(buf)

    assert type(restored) is type(T)
    assert restored.shape == T.shape
    for i in range(T.shape[0]):
        np.testing.assert_array_equal(T[i].to_dense(), restored[i].to_dense())


@pytest.mark.parametrize("symmat_dtype, scalar_dtype", [
    (mpcf.symmat32, mpcf.float32),
    (mpcf.symmat64, mpcf.float64),
])
def test_symmetric_matrix_tensor_roundtrip_empty(symmat_dtype, scalar_dtype):
    T = mpcf.zeros((1,), dtype=symmat_dtype)
    T[0] = mpcf.SymmetricMatrix(0, dtype=scalar_dtype)

    buf = io.BytesIO()
    mpcf.save(T, buf)

    buf.seek(0)
    restored = mpcf.load(buf)

    assert type(restored) is type(T)
    assert restored.shape == T.shape
    assert restored[0].size == 0


def _make_distance_matrix(n, dtype):
    mat = mpcf.DistanceMatrix(n, dtype=dtype)
    for i in range(n):
        for j in range(i):
            mat[i, j] = float(i * n + j + 1)
    return mat


@pytest.mark.parametrize("distmat_dtype, scalar_dtype", [
    (mpcf.distmat32, mpcf.float32),
    (mpcf.distmat64, mpcf.float64),
])
def test_distance_matrix_tensor_roundtrip(distmat_dtype, scalar_dtype):
    T = mpcf.zeros((2,), dtype=distmat_dtype)
    T[0] = _make_distance_matrix(3, scalar_dtype)
    T[1] = _make_distance_matrix(4, scalar_dtype)

    buf = io.BytesIO()
    mpcf.save(T, buf)

    buf.seek(0)
    restored = mpcf.load(buf)

    assert type(restored) is type(T)
    assert restored.shape == T.shape
    for i in range(T.shape[0]):
        np.testing.assert_array_equal(T[i].to_dense(), restored[i].to_dense())


@pytest.mark.parametrize("distmat_dtype, scalar_dtype", [
    (mpcf.distmat32, mpcf.float32),
    (mpcf.distmat64, mpcf.float64),
])
def test_distance_matrix_tensor_roundtrip_empty(distmat_dtype, scalar_dtype):
    T = mpcf.zeros((1,), dtype=distmat_dtype)
    T[0] = mpcf.DistanceMatrix(0, dtype=scalar_dtype)

    buf = io.BytesIO()
    mpcf.save(T, buf)

    buf.seek(0)
    restored = mpcf.load(buf)

    assert type(restored) is type(T)
    assert restored.shape == T.shape
    assert restored[0].size == 0


# --- TimeSeries ---


@pytest.mark.parametrize("ts_dtype", [mpcf.ts32, mpcf.ts64])
def test_timeseries_tensor_roundtrip(ts_dtype):
    np_dtype = np.float32 if ts_dtype == mpcf.ts32 else np.float64
    ts1 = TimeSeries(np.array([1.0, 2.0, 3.0], dtype=np_dtype),
                     start_time=0.0, time_step=1.0, dtype=ts_dtype)
    ts2 = TimeSeries(np.array([4.0, 5.0, 6.0], dtype=np_dtype),
                     start_time=10.0, time_step=0.5, dtype=ts_dtype)
    T = TimeSeriesTensor([ts1, ts2])

    buf = io.BytesIO()
    mpcf.save(T, buf)

    buf.seek(0)
    restored = mpcf.load(buf)

    assert type(restored) is type(T)
    assert restored.dtype == T.dtype
    assert restored.shape == T.shape
    assert T.array_equal(restored)


def test_timeseries_object_roundtrip():
    ts = TimeSeries(np.array([10.0, 20.0, 30.0]),
                    start_time=5.0, time_step=0.25)

    buf = io.BytesIO()
    mpcf.save(ts, buf)

    buf.seek(0)
    restored = mpcf.load(buf)

    assert isinstance(restored, TimeSeries)
    assert restored == ts
