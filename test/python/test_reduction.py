import pytest
import masspcf as mpcf
import pytest

import numpy as np

def test_mean_of_1d_returns_pcf():
    A = mpcf.zeros((10,))
    avg = mpcf.mean(A)

    assert isinstance(avg, mpcf.Pcf)
    assert np.array_equal(avg.to_numpy(), np.zeros((1,2)))


def test_mean_of_simple():
    A = mpcf.zeros((2,))

    A[0] = mpcf.Pcf(np.array([[0., 1.], [2., 3.]], dtype=np.float32))

    avg = mpcf.mean(A)
    avg_data = avg.to_numpy()

    assert avg_data.shape == (2, 2)
    assert avg_data[0][0] == pytest.approx(0., 1e-4)
    assert avg_data[0][1] == pytest.approx(0.5, 1e-4)
    assert avg_data[1][0] == pytest.approx(2., 1e-4)
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
