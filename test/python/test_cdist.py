"""Tests for cdist (cross-distance computation)."""

import itertools

import numpy as np
import pytest

import masspcf as mpcf


def assert_cdist_matches_lp_distance(D, X, Y, p=1):
    """Assert every entry of cdist result D matches lp_distance(X[xi], Y[yi])."""
    x_ranges = [range(s) for s in X.shape]
    y_ranges = [range(s) for s in Y.shape]

    for xi in itertools.product(*x_ranges):
        for yi in itertools.product(*y_ranges):
            idx = xi + yi
            assert D[idx] == pytest.approx(mpcf.lp_distance(X[xi], Y[yi], p=p)), \
                f"Mismatch at D{list(idx)}"


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
    assert_cdist_matches_lp_distance(D, X, Y)


def test_cdist_multidim():
    X = mpcf.zeros((2, 3), dtype=mpcf.pcf64)
    Y = mpcf.zeros((4,), dtype=mpcf.pcf64)

    for i in range(2):
        for j in range(3):
            X[i, j] = mpcf.Pcf(np.array([[0.0, float(i * 3 + j + 1)], [1.0, 0.0]]))

    for k in range(4):
        Y[k] = mpcf.Pcf(np.array([[0.0, float(k)], [1.0, 0.0]]))

    D = mpcf.cdist(X, Y)

    assert D.shape == (2, 3, 4)
    assert_cdist_matches_lp_distance(D, X, Y)


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
    assert_cdist_matches_lp_distance(D, X, Y, p=3)


def test_cdist_both_multidim():
    X = mpcf.zeros((2, 3), dtype=mpcf.pcf64)
    Y = mpcf.zeros((4, 2), dtype=mpcf.pcf64)

    for i in range(2):
        for j in range(3):
            X[i, j] = mpcf.Pcf(np.array([[0.0, float(i * 3 + j + 1)], [1.0, 0.0]]))

    for i in range(4):
        for j in range(2):
            Y[i, j] = mpcf.Pcf(np.array([[0.0, float(i * 2 + j)], [1.0, 0.0]]))

    D = mpcf.cdist(X, Y)

    assert D.shape == (2, 3, 4, 2)
    assert_cdist_matches_lp_distance(D, X, Y)


def test_cdist_sliced_inputs():
    """cdist with noncontiguous (sliced) tensor views."""
    X_full = mpcf.zeros((6,), dtype=mpcf.pcf64)
    Y_full = mpcf.zeros((8,), dtype=mpcf.pcf64)

    for i in range(6):
        X_full[i] = mpcf.Pcf(np.array([[0.0, float(i + 1)], [1.0, 0.0]]))
    for i in range(8):
        Y_full[i] = mpcf.Pcf(np.array([[0.0, float(i)], [1.0, 0.0]]))

    X = X_full[::2]  # elements 0, 2, 4 => values 1, 3, 5
    Y = Y_full[1::3]  # elements 1, 4, 7 => values 1, 4, 7

    D = mpcf.cdist(X, Y)

    assert D.shape == (3, 3)
    assert_cdist_matches_lp_distance(D, X, Y)


def test_cdist_sliced_multidim():
    """cdist with noncontiguous slices of multidimensional tensors."""
    X_full = mpcf.zeros((4, 6), dtype=mpcf.pcf64)
    Y_full = mpcf.zeros((3, 8), dtype=mpcf.pcf64)

    for i in range(4):
        for j in range(6):
            X_full[i, j] = mpcf.Pcf(np.array([[0.0, float(i * 6 + j + 1)], [1.0, 0.0]]))
    for i in range(3):
        for j in range(8):
            Y_full[i, j] = mpcf.Pcf(np.array([[0.0, float(i * 8 + j)], [1.0, 0.0]]))

    X = X_full[::2, 1::2]  # shape (2, 3)
    Y = Y_full[:, ::3]     # shape (3, 3)

    assert X.shape == (2, 3)
    assert Y.shape == (3, 3)

    D = mpcf.cdist(X, Y)

    assert D.shape == (2, 3, 3, 3)
    assert_cdist_matches_lp_distance(D, X, Y)


def test_cdist_empty():
    X = mpcf.zeros((0,), dtype=mpcf.pcf64)
    Y = mpcf.zeros((3,), dtype=mpcf.pcf64)

    D = mpcf.cdist(X, Y)
    assert D.shape == (0, 3)
