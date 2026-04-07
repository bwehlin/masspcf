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


def test_pdist_l3_constant_pcfs():
    X = mpcf.zeros((2,), dtype=mpcf.pcf64)

    # f(t) = 4 on [0, 1), g(t) = 1 on [0, 1)
    X[0] = mpcf.Pcf(np.array([[0.0, 4.0], [1.0, 0.0]]))
    X[1] = mpcf.Pcf(np.array([[0.0, 1.0], [1.0, 0.0]]))

    D = mpcf.pdist(X, p=3)

    assert isinstance(D, DistanceMatrix)
    assert D.size == 2
    # ||f - g||_3 = (integral |4 - 1|^3 dt)^(1/3) = (27)^(1/3) = 3
    assert D[0, 1] == pytest.approx(3.0)
    assert D[0, 0] == 0.0
    assert D[1, 1] == 0.0


def test_pdist_lp_returns_distance_matrix():
    X = mpcf.zeros((3,))
    D = mpcf.pdist(X, p=3)
    assert isinstance(D, DistanceMatrix)


def test_from_dense_valid():
    dense = np.array([[0.0, 1.0, 2.0],
                       [1.0, 0.0, 3.0],
                       [2.0, 3.0, 0.0]])
    dm = DistanceMatrix.from_dense(dense)
    assert dm.size == 3
    assert dm[0, 1] == 1.0
    assert dm[0, 2] == 2.0
    assert dm[1, 2] == 3.0


def test_from_dense_rejects_nonzero_diagonal():
    dense = np.array([[1.0, 0.0],
                       [0.0, 0.0]])
    with pytest.raises(ValueError, match="Diagonal"):
        DistanceMatrix.from_dense(dense)


def test_from_dense_rejects_negative():
    dense = np.array([[0.0, -1.0],
                       [-1.0, 0.0]])
    with pytest.raises(ValueError, match="nonnegative"):
        DistanceMatrix.from_dense(dense)


def test_from_dense_rejects_asymmetric():
    dense = np.array([[0.0, 1.0],
                       [2.0, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        DistanceMatrix.from_dense(dense)


# --- Tests for PcfContainerLike acceptance (list / single Pcf) ---


def test_pdist_accepts_list_of_pcfs():
    f = mpcf.Pcf(np.array([[0.0, 10.0], [2.0, 5.0], [3.0, 0.0]]))
    g = mpcf.Pcf(np.array([[0.0, 5.0], [6.0, 0.0]]))

    D = mpcf.pdist([f, g], verbose=False)

    assert isinstance(D, DistanceMatrix)
    assert D.size == 2
    assert D[0, 0] == 0.0
    assert D[0, 1] == pytest.approx(2 * 5 + 3 * 5)


def test_cdist_accepts_lists_of_pcfs():
    f = mpcf.Pcf(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))
    g = mpcf.Pcf(np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float64))

    D = mpcf.cdist([f], [g], verbose=False)

    assert D.shape == (1, 1)
    assert float(np.asarray(D).flat[0]) == pytest.approx(1.0)


def test_lp_norm_accepts_list_of_pcfs():
    f = mpcf.Pcf(np.array([[0.0, 3.0], [1.0, 0.0]], dtype=np.float64))

    norms = mpcf.lp_norm([f], verbose=False)

    assert norms.shape == (1,)
    assert norms[0] == pytest.approx(3.0)


def test_pdist_accepts_empty_list():
    D = mpcf.pdist([], verbose=False)

    assert isinstance(D, DistanceMatrix)
    assert D.size == 0
