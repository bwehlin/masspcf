"""Tests for cdist (cross-distance computation)."""

import numpy as np
import pytest

import masspcf as mpcf


def test_cdist_rejects_p_less_than_1():
    X = mpcf.zeros((2,))
    Y = mpcf.zeros((3,))
    with pytest.raises(ValueError, match="p must be >= 1"):
        mpcf.cdist(X, Y, p=0.5)


def test_cdist_1d_tensors():
    X = mpcf.zeros((2,), dtype=mpcf.pcf64)
    Y = mpcf.zeros((3,), dtype=mpcf.pcf64)

    X[0] = mpcf.Pcf(np.array([[0.0, 5.0], [1.0, 0.0]]))
    X[1] = mpcf.Pcf(np.array([[0.0, 1.0], [2.0, 0.0]]))

    Y[0] = mpcf.Pcf(np.array([[0.0, 0.0]]))
    Y[1] = mpcf.Pcf(np.array([[0.0, 5.0], [1.0, 0.0]]))
    Y[2] = mpcf.Pcf(np.array([[0.0, 1.0]]))

    D = mpcf.cdist(X, Y)

    assert isinstance(D, mpcf.FloatTensor)
    assert D.shape == (2, 3)

    # X[0] = 5 on [0,1), 0 on [1,inf)
    # Y[0] = 0 everywhere => ||X[0]-Y[0]||_1 = 5
    assert D[0, 0] == pytest.approx(5.0)
    # X[0] vs Y[1] = same function => distance 0
    assert D[0, 1] == pytest.approx(0.0)


def test_cdist_output_shape_multidim():
    X = mpcf.zeros((2, 3), dtype=mpcf.pcf64)
    Y = mpcf.zeros((4,), dtype=mpcf.pcf64)

    D = mpcf.cdist(X, Y)

    assert D.shape == (2, 3, 4)


def test_cdist_matches_pdist_when_same_input():
    X = mpcf.zeros((3,), dtype=mpcf.pcf64)

    X[0] = mpcf.Pcf(np.array([[0.0, 3.0], [1.0, 0.0]]))
    X[1] = mpcf.Pcf(np.array([[0.0, 1.0], [2.0, 0.0]]))
    X[2] = mpcf.Pcf(np.array([[0.0, 0.0]]))

    D_pdist = mpcf.pdist(X)
    D_cdist = mpcf.cdist(X, X)

    assert D_cdist.shape == (3, 3)

    for i in range(3):
        for j in range(3):
            assert D_cdist[i, j] == pytest.approx(D_pdist[i, j], abs=1e-10), \
                f"Mismatch at ({i}, {j})"


def test_cdist_lp():
    X = mpcf.zeros((2,), dtype=mpcf.pcf64)
    Y = mpcf.zeros((1,), dtype=mpcf.pcf64)

    X[0] = mpcf.Pcf(np.array([[0.0, 4.0], [1.0, 0.0]]))
    X[1] = mpcf.Pcf(np.array([[0.0, 1.0], [1.0, 0.0]]))
    Y[0] = mpcf.Pcf(np.array([[0.0, 1.0], [1.0, 0.0]]))

    D = mpcf.cdist(X, Y, p=3)

    assert D.shape == (2, 1)
    # ||X[0] - Y[0]||_3 = (|4-1|^3 * 1)^(1/3) = 3
    assert D[0, 0] == pytest.approx(3.0)
    # ||X[1] - Y[0]||_3 = 0
    assert D[1, 0] == pytest.approx(0.0)


def test_cdist_empty():
    X = mpcf.zeros((0,), dtype=mpcf.pcf64)
    Y = mpcf.zeros((3,), dtype=mpcf.pcf64)

    D = mpcf.cdist(X, Y)
    assert D.shape == (0, 3)
