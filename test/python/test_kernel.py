import numpy as np
import pytest

import masspcf as mpcf
from masspcf.symmetric_matrix import SymmetricMatrix


def test_l2_kernel_requires_1d_tensor():
    X = mpcf.zeros((10, 20))

    with pytest.raises(ValueError):
        mpcf.l2_kernel(X)

    mpcf.l2_kernel(X[:, 2])


def test_l2_kernel_returns_symmetric_matrix():
    X = mpcf.zeros((3,))
    K = mpcf.l2_kernel(X)
    assert isinstance(K, SymmetricMatrix)


def test_l2_kernel_of_empty():
    X = mpcf.zeros((0,))
    K = mpcf.l2_kernel(X)

    assert isinstance(K, SymmetricMatrix)
    assert K.size == 0


def test_l2_kernel_of_one(pcf_dtype, device):
    np_dtype = np.float32 if pcf_dtype == mpcf.pcf32 else np.float64
    X = mpcf.zeros((1,), dtype=pcf_dtype)
    X[0] = mpcf.Pcf(np.array([[0.0, 3.0], [2.0, 0.0]], dtype=np_dtype))

    K = mpcf.l2_kernel(X)

    assert K.size == 1
    # <f, f> = integral of f(t)^2 = 3^2 * 2 = 18
    assert K[0, 0] == pytest.approx(18.0, rel=1e-5)


def test_l2_kernel_of_two(pcf_dtype, device):
    np_dtype = np.float32 if pcf_dtype == mpcf.pcf32 else np.float64
    X = mpcf.zeros((2,), dtype=pcf_dtype)

    # f(t) = 4 on [0, 3)
    X[0] = mpcf.Pcf(np.array([[0.0, 4.0], [3.0, 0.0]], dtype=np_dtype))
    # g(t) = 2 on [0, 3)
    X[1] = mpcf.Pcf(np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np_dtype))

    K = mpcf.l2_kernel(X)

    assert K.size == 2

    # <f, f> = 4*4*3 = 48
    assert K[0, 0] == pytest.approx(48.0, rel=1e-5)
    # <f, g> = 4*2*3 = 24
    assert K[0, 1] == pytest.approx(24.0, rel=1e-5)
    assert K[1, 0] == K[0, 1]
    # <g, g> = 2*2*3 = 12
    assert K[1, 1] == pytest.approx(12.0, rel=1e-5)


def test_l2_kernel_to_dense(pcf_dtype, device):
    np_dtype = np.float32 if pcf_dtype == mpcf.pcf32 else np.float64
    X = mpcf.zeros((2,), dtype=pcf_dtype)

    X[0] = mpcf.Pcf(np.array([[0.0, 4.0], [3.0, 0.0]], dtype=np_dtype))
    X[1] = mpcf.Pcf(np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np_dtype))

    K = mpcf.l2_kernel(X)
    dense = K.to_dense()

    assert isinstance(dense, np.ndarray)
    assert dense.shape == (2, 2)
    assert dense[0, 0] == pytest.approx(48.0, rel=1e-5)
    assert dense[0, 1] == pytest.approx(24.0, rel=1e-5)
    assert dense[1, 0] == dense[0, 1]
    assert dense[1, 1] == pytest.approx(12.0, rel=1e-5)
