import pytest
import masspcf as mpcf

import numpy as np

def test_mean_of_1d_returns_pcf():
    A = mpcf.zeros((10,))
    avg = mpcf.mean(A)

    assert isinstance(avg, mpcf.Pcf)
    assert np.array_equal(avg.to_numpy(), np.zeros((1,2)))

