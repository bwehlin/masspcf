import masspcf as mpcf
import numpy as np
import numpy.testing as npt

def get_test_data():
    X = mpcf.zeros((2,3), dtype=mpcf.float64)

    X[0, 0] = mpcf.Pcf(np.array([[0., 10.], [2., 5.], [4., 0.]]))
    X[0, 1] = mpcf.Pcf(np.array([[0., 5.], [3., 6.], [5., 5.], [8., 0.]]))
    X[0, 2] = mpcf.Pcf(np.array([[0., 12.], [3., 0.]]))

    X[1, 0] = mpcf.Pcf(np.array([[0., 50.], [0.5, 0.]]))
    X[1, 1] = mpcf.Pcf(np.array([[0., 0.]]))
    X[1, 2] = mpcf.Pcf(np.array([[0., 2.], [3., 0.]]))

    return X

def test_l1_norm():
    X = get_test_data()

    expected = np.zeros((2, 3))

    expected[0, 0] = 2 * 10 + 2 * 5
    expected[0, 1] = 3 * 5 + 2 * 6 + 3 * 5
    expected[0, 2] = 3 * 12

    expected[1, 0] = 0.5 * 50
    expected[1, 1] = 0
    expected[1, 2] = 3 * 2

    actual = mpcf.lp_norm(X, p=1)

    assert isinstance(actual, np.ndarray)
    assert actual.shape == (2, 3)

    npt.assert_allclose(actual, expected, rtol=1e-7, atol=0)
