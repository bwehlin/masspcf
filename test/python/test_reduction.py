import numpy as np
import pytest

import masspcf as mpcf


def test_mean_of_1d_returns_pcf():
    A = mpcf.zeros((10,))
    avg = mpcf.mean(A)

    assert isinstance(avg, mpcf.Pcf)
    assert np.array_equal(avg.to_numpy(), np.zeros((1, 2)))


def test_mean_of_simple():
    A = mpcf.zeros((2,))

    A[0] = mpcf.Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))

    avg = mpcf.mean(A)
    avg_data = avg.to_numpy()

    assert avg_data.shape == (2, 2)
    assert avg_data[0][0] == pytest.approx(0.0, 1e-4)
    assert avg_data[0][1] == pytest.approx(0.5, 1e-4)
    assert avg_data[1][0] == pytest.approx(2.0, 1e-4)
    assert avg_data[1][1] == pytest.approx(1.5, 1e-4)


"""
def test_mean_of_2x3():
    X = mpcf.zeros((2,3), dtype=mpcf.float64)

    X[0, 0] = mpcf.Pcf(np.array([[0., 10.], [2., 5.], [4., 0.]]))
    X[0, 1] = mpcf.Pcf(np.array([[0., 5.], [3., 6.], [5., 5.], [8., 0.]]))
    X[0, 2] = mpcf.Pcf(np.array([[0., 12.], [3., 0.]]))

    X[1, 0] = mpcf.Pcf(np.array([[0., 50.], [0.5, 0.]]))
    X[1, 1] = mpcf.Pcf(np.array([[0., 0.]]))
    X[1, 2] = mpcf.Pcf(np.array([[0., 2.], [3., 0.]]))

    avg = mpcf.mean(X)
    avg0 = mpcf.mean(X, dim=0)
    avg1 = mpcf.mean(X, dim=1)

    assert avg.shape == (3)
    assert avg0.shape == (3)
    assert avg1.shape == (2)

    assert False
"""


def test_mean_f64():
    A = mpcf.zeros((2,), dtype=mpcf.pcf64)
    A[0] = mpcf.Pcf(np.array([[0.0, 4.0], [2.0, 0.0]], dtype=np.float64))
    A[1] = mpcf.Pcf(np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64))

    avg = mpcf.mean(A)
    assert isinstance(avg, mpcf.Pcf)
    avg_data = avg.to_numpy()
    assert avg_data[0][1] == pytest.approx(3.0, abs=1e-7)


def test_mean_2d_dim0():
    X = mpcf.zeros((2, 3))
    X[0, 0] = mpcf.Pcf(np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32))
    X[1, 0] = mpcf.Pcf(np.array([[0.0, 4.0], [1.0, 0.0]], dtype=np.float32))

    avg = mpcf.mean(X, dim=0)
    assert avg.shape == (3,)


def test_mean_2d_dim1():
    X = mpcf.zeros((2, 3))
    X[0, 0] = mpcf.Pcf(np.array([[0.0, 6.0], [1.0, 0.0]], dtype=np.float32))
    X[0, 1] = mpcf.Pcf(np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32))
    X[0, 2] = mpcf.Pcf(np.array([[0.0, 4.0], [1.0, 0.0]], dtype=np.float32))

    avg = mpcf.mean(X, dim=1)
    assert avg.shape == (2,)


def test_get_tensor_and_backend_invalid():
    from masspcf.reductions import _get_tensor_and_backend

    with pytest.raises(ValueError, match="not supported"):
        _get_tensor_and_backend("not a tensor")


def test_max_time_1d():
    A = mpcf.zeros((2,))
    A[0] = mpcf.Pcf(np.array([[0.0, 1.0], [5.0, 0.0]], dtype=np.float32))
    A[1] = mpcf.Pcf(np.array([[0.0, 1.0], [3.0, 0.0]], dtype=np.float32))

    result = mpcf.max_time(A)
    assert isinstance(result, float)
    assert result == pytest.approx(5.0, abs=1e-5)


def test_max_time_f64():
    A = mpcf.zeros((2,), dtype=mpcf.pcf64)
    A[0] = mpcf.Pcf(np.array([[0.0, 1.0], [7.0, 0.0]], dtype=np.float64))
    A[1] = mpcf.Pcf(np.array([[0.0, 1.0], [3.0, 0.0]], dtype=np.float64))

    result = mpcf.max_time(A)
    assert isinstance(result, float)
    assert result == pytest.approx(7.0, abs=1e-10)


def test_max_time_2d():
    X = mpcf.zeros((2, 3))
    X[0, 0] = mpcf.Pcf(np.array([[0.0, 1.0], [10.0, 0.0]], dtype=np.float32))
    X[0, 1] = mpcf.Pcf(np.array([[0.0, 1.0], [6.0, 0.0]], dtype=np.float32))
    X[0, 2] = mpcf.Pcf(np.array([[0.0, 1.0], [4.0, 0.0]], dtype=np.float32))
    X[1, 0] = mpcf.Pcf(np.array([[0.0, 1.0], [3.0, 0.0]], dtype=np.float32))
    X[1, 1] = mpcf.Pcf(np.array([[0.0, 1.0], [8.0, 0.0]], dtype=np.float32))
    X[1, 2] = mpcf.Pcf(np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32))

    result = mpcf.max_time(X, dim=0)
    assert result.shape == (3,)
    result_np = np.asarray(result)
    assert result_np[0] == pytest.approx(10.0, abs=1e-5)
    assert result_np[1] == pytest.approx(8.0, abs=1e-5)
    assert result_np[2] == pytest.approx(4.0, abs=1e-5)

    result1 = mpcf.max_time(X, dim=1)
    assert result1.shape == (2,)
    result1_np = np.asarray(result1)
    assert result1_np[0] == pytest.approx(10.0, abs=1e-5)
    assert result1_np[1] == pytest.approx(8.0, abs=1e-5)
