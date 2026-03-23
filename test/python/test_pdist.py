import numpy as np
import pytest

import masspcf as mpcf
from masspcf.distance_matrix import DistanceMatrix


def test_pdist_rejects_p_less_than_1():
    X = mpcf.zeros((2,))
    with pytest.raises(ValueError, match="p must be >= 1"):
        mpcf.pdist(X, p=0.5)


def test_pdist_rejects_p_zero():
    X = mpcf.zeros((2,))
    with pytest.raises(ValueError, match="p must be >= 1"):
        mpcf.pdist(X, p=0)


def test_pdist_rejects_p_negative():
    X = mpcf.zeros((2,))
    with pytest.raises(ValueError, match="p must be >= 1"):
        mpcf.pdist(X, p=-1)


def test_pdist_requires_1d_tensor():
    X = mpcf.zeros((10, 20))

    with pytest.raises(ValueError):
        mpcf.pdist(X)

    mpcf.pdist(X[:, 2])


def test_pdist_returns_distance_matrix():
    X = mpcf.zeros((3,))
    D = mpcf.pdist(X)
    assert isinstance(D, DistanceMatrix)


def test_pdist_of_empty_gives_empty():
    X = mpcf.zeros((0,))
    D = mpcf.pdist(X)

    assert isinstance(D, DistanceMatrix)
    assert D.size == 0


def test_pdist_of_one_gives_zero_1x1():
    X = mpcf.zeros((1,))
    D = mpcf.pdist(X)

    assert D.size == 1
    assert D[0, 0] == 0.0


def test_pdist_of_two_gives_correct_output():
    X = mpcf.zeros((2,), dtype=mpcf.pcf64)

    X[0] = mpcf.Pcf(np.array([[0.0, 10.0], [2.0, 5.0], [3.0, 0.0]]))
    X[1] = mpcf.Pcf(np.array([[0.0, 5.0], [6.0, 0.0]]))

    D = mpcf.pdist(X)

    assert D.size == 2

    assert D[0, 0] == 0.0
    assert D[0, 1] == pytest.approx(2 * 5 + 3 * 5)
    assert D[1, 0] == D[0, 1]
    assert D[1, 1] == 0.0


def test_pdist_to_dense():
    X = mpcf.zeros((2,), dtype=mpcf.pcf64)

    X[0] = mpcf.Pcf(np.array([[0.0, 10.0], [2.0, 5.0], [3.0, 0.0]]))
    X[1] = mpcf.Pcf(np.array([[0.0, 5.0], [6.0, 0.0]]))

    D = mpcf.pdist(X)
    dense = D.to_dense()

    assert isinstance(dense, np.ndarray)
    assert dense.shape == (2, 2)
    assert dense[0, 0] == 0.0
    assert dense[0, 1] == pytest.approx(2 * 5 + 3 * 5)
    assert dense[1, 0] == dense[0, 1]
    assert dense[1, 1] == 0.0
