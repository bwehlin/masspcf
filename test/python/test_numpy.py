import pytest
import masspcf as mpcf
import numpy as np

def test_conversion():
    Xnp = np.array([[0., 1.], [2., 5.], [6., 0.]])
    Xpcf = mpcf.Pcf(Xnp)

    Xconv = np.array(Xpcf)

    assert(np.array_equal(Xnp, Xconv))

def test_shape_conversion():
    X = mpcf.zeros((10, 3, 2))

    Y = np.zeros(X.shape)

    assert(Y.shape[0] == 10)
    assert(Y.shape[1] == 3)
    assert(Y.shape[2] == 2)
