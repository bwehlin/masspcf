import pytest
import masspcf as mpcf
import numpy as np

def test_conversion():
    Xnp = np.array([[0., 1.], [2., 5.], [6., 0.]])
    Xpcf = mpcf.Pcf(Xnp)

    Xconv = np.array(Xpcf)

    assert(np.array_equal(Xnp, Xconv))
