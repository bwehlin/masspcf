import masspcf as mpcf
import pytest

import numpy as np

def test_pdist_requires_1d_tensor():
    X = mpcf.zeros((10, 20))

    with pytest.raises(ValueError):
        mpcf.pdist(X)

    mpcf.pdist(X[:, 2])

def test_pdist_of_empty_gives_empty():
    X = mpcf.zeros((0,))
    D = mpcf.pdist(X)

    assert len(D.shape) == 2
    assert D.shape[0] == 0
    assert D.shape[1] == 0

def test_pdist_of_one_gives_zero_1x1():
    X = mpcf.zeros((1,))
    D = mpcf.pdist(X)

    assert len(D.shape) == 2
    assert D.shape[0] == 1
    assert D.shape[1] == 1

    assert D[0, 0] == 0.

def test_pdist_of_two_gives_correct_output():
    X = mpcf.zeros((2,), dtype=mpcf.pcf64)

    X[0] = mpcf.Pcf(np.array([[0., 10.], [2., 5.], [3., 0.]]))
    X[1] = mpcf.Pcf(np.array([[0., 5.], [6., 0.]]))

    D = mpcf.pdist(X)

    assert D.dtype == np.float64

    assert len(D.shape) == 2
    assert D.shape[0] == 2
    assert D.shape[1] == 2

    assert D[0, 0] == 0.
    assert D[0, 1] == pytest.approx(2 * 5 + 3 * 5)
    assert D[1, 0] == D[0, 1]
    assert D[1, 1] == 0.
