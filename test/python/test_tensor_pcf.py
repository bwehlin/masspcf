import numpy as np
import numpy.testing as npt

import masspcf as mpcf


def test_pcf_tensor_basic():
    X = mpcf.zeros((2, 2))

    assert X.dtype == mpcf.pcf32

    data = np.array([[0, 3], [1, 5], [2, 0]])

    X[0, 0] = mpcf.Pcf(data, dtype=mpcf.pcf32)

    X00np = np.asarray(X[0, 0])

    npt.assert_equal(data, X00np)
