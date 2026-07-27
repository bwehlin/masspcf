import io
from pathlib import Path

import numpy as np
import pytest

import stablebear as sb


def test_float32_tensor_roundtrip():
    original = sb.FloatTensor(np.random.randn(2, 3))

    buf = io.BytesIO()
    sb.save(original, buf)

    buf.seek(0)  # rewind before reading!
    restored = sb.load(buf)

    assert original.array_equal(restored)


def _make_symmetric_matrix(n, dtype):
    mat = sb.SymmetricMatrix(n, dtype=dtype)
    for i in range(n):
        for j in range(i + 1):
            mat[i, j] = float(i * n + j)
    return mat


@pytest.mark.parametrize("symmat_dtype, scalar_dtype", [
    (sb.symmat32, sb.float32),
    (sb.symmat64, sb.float64),
])
def test_symmetric_matrix_tensor_roundtrip(symmat_dtype, scalar_dtype):
    T = sb.zeros((2,), dtype=symmat_dtype)
    T[0] = _make_symmetric_matrix(3, scalar_dtype)
    T[1] = _make_symmetric_matrix(4, scalar_dtype)

    buf = io.BytesIO()
    sb.save(T, buf)

    buf.seek(0)
    restored = sb.load(buf)

    assert type(restored) is type(T)
    assert restored.shape == T.shape
    for i in range(T.shape[0]):
        np.testing.assert_array_equal(T[i].to_dense(), restored[i].to_dense())


@pytest.mark.parametrize("symmat_dtype, scalar_dtype", [
    (sb.symmat32, sb.float32),
    (sb.symmat64, sb.float64),
])
def test_symmetric_matrix_tensor_roundtrip_empty(symmat_dtype, scalar_dtype):
    T = sb.zeros((1,), dtype=symmat_dtype)
    T[0] = sb.SymmetricMatrix(0, dtype=scalar_dtype)

    buf = io.BytesIO()
    sb.save(T, buf)

    buf.seek(0)
    restored = sb.load(buf)

    assert type(restored) is type(T)
    assert restored.shape == T.shape
    assert restored[0].size == 0


def _make_distance_matrix(n, dtype):
    mat = sb.DistanceMatrix(n, dtype=dtype)
    for i in range(n):
        for j in range(i):
            mat[i, j] = float(i * n + j + 1)
    return mat


@pytest.mark.parametrize("distmat_dtype, scalar_dtype", [
    (sb.distmat32, sb.float32),
    (sb.distmat64, sb.float64),
])
def test_distance_matrix_tensor_roundtrip(distmat_dtype, scalar_dtype):
    T = sb.zeros((2,), dtype=distmat_dtype)
    T[0] = _make_distance_matrix(3, scalar_dtype)
    T[1] = _make_distance_matrix(4, scalar_dtype)

    buf = io.BytesIO()
    sb.save(T, buf)

    buf.seek(0)
    restored = sb.load(buf)

    assert type(restored) is type(T)
    assert restored.shape == T.shape
    for i in range(T.shape[0]):
        np.testing.assert_array_equal(T[i].to_dense(), restored[i].to_dense())


@pytest.mark.parametrize("distmat_dtype, scalar_dtype", [
    (sb.distmat32, sb.float32),
    (sb.distmat64, sb.float64),
])
def test_distance_matrix_tensor_roundtrip_empty(distmat_dtype, scalar_dtype):
    T = sb.zeros((1,), dtype=distmat_dtype)
    T[0] = sb.DistanceMatrix(0, dtype=scalar_dtype)

    buf = io.BytesIO()
    sb.save(T, buf)

    buf.seek(0)
    restored = sb.load(buf)

    assert type(restored) is type(T)
    assert restored.shape == T.shape
    assert restored[0].size == 0


# Legacy-format fixtures: bytes written by an older stablebear (main's
# pre-indexed-views writer), checked into test/data/legacy_io/. The expected
# values mirror generate_fixtures.cpp in that directory — keep them in sync.

_LEGACY_DIR = Path(__file__).resolve().parent.parent / "data" / "legacy_io"


def _legacy_pcloud(c):
    return np.array([[100 * c + 10 * i + j + 0.25 for j in range(2)]
                     for i in range(3 - c)])


def _legacy_distmat_dense(n, offset):
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = 10 * i + j + offset
    return D


@pytest.mark.parametrize("suffix, np_dtype, pcloud_dtype", [
    ("f32", np.float32, sb.pcloud32),
    ("f64", np.float64, sb.pcloud64),
])
def test_load_legacy_point_cloud_tensor_fixture(suffix, np_dtype, pcloud_dtype):
    t = sb.load(str(_LEGACY_DIR / f"pcloud_tensor_{suffix}.sb"))

    assert isinstance(t, sb.PointCloudTensor)
    assert t.dtype == pcloud_dtype
    assert t.shape == (2,)
    for c in range(2):
        np.testing.assert_array_equal(
            t[c].to_numpy(), _legacy_pcloud(c).astype(np_dtype))


@pytest.mark.parametrize("suffix, np_dtype, distmat_dtype", [
    ("f32", np.float32, sb.distmat32),
    ("f64", np.float64, sb.distmat64),
])
def test_load_legacy_distance_matrix_tensor_fixture(suffix, np_dtype, distmat_dtype):
    t = sb.load(str(_LEGACY_DIR / f"distmat_tensor_{suffix}.sb"))

    assert isinstance(t, sb.DistanceMatrixTensor)
    assert t.dtype == distmat_dtype
    assert t.shape == (2,)
    for c in range(2):
        np.testing.assert_array_equal(
            t[c].to_dense(), _legacy_distmat_dense(3 + c, 0.5).astype(np_dtype))


@pytest.mark.parametrize("suffix, np_dtype, scalar_dtype", [
    ("f32", np.float32, sb.float32),
    ("f64", np.float64, sb.float64),
])
def test_load_legacy_distance_matrix_object_fixture(suffix, np_dtype, scalar_dtype):
    m = sb.load(str(_LEGACY_DIR / f"distmat_object_{suffix}.sb"))

    assert isinstance(m, sb.DistanceMatrix)
    assert m.dtype == scalar_dtype
    np.testing.assert_array_equal(
        m.to_dense(), _legacy_distmat_dense(5, 0.25).astype(np_dtype))
