import pytest
import masspcf as mpcf

import numpy as np
import numpy.testing as npt

def test_pcf_tensor_basic():
    X = mpcf.zeros((2, 2))

    assert X.dtype == mpcf.float32
    assert X._type == mpcf.tensor.TensorType.PCF

    data = np.array([[0, 3], [1, 5], [2, 0]])

    X[0, 0] = mpcf.Pcf(data, dtype=mpcf.float32)

    X00np = np.array(X[0, 0])
    
    npt.assert_equal(data, X00np)



