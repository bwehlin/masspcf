import numpy as np

import masspcf as mpcf
import masspcf._mpcf_cpp as cpp


def test_np_to_pcf_has_correct_type():
    X32 = np.zeros((2, 2), dtype=np.float32)
    X64 = np.zeros((2, 2), dtype=np.float64)

    f32 = mpcf.Pcf(X32)
    assert isinstance(f32._data, cpp.Pcf_f32_f32)
    assert f32.ttype == mpcf.f32
    assert f32.vtype == mpcf.f32

    f64 = mpcf.Pcf(X64)
    assert isinstance(f64._data, cpp.Pcf_f64_f64)
    assert f64.ttype == mpcf.f64
    assert f64.vtype == mpcf.f64

    f32_64 = mpcf.Pcf(X32, dtype=mpcf.pcf64)
    assert isinstance(f32_64._data, cpp.Pcf_f64_f64)
    assert f32_64.ttype == mpcf.f64
    assert f32_64.vtype == mpcf.f64

    f64_32 = mpcf.Pcf(X64, dtype=mpcf.pcf32)
    assert isinstance(f64_32._data, cpp.Pcf_f32_f32)
    assert f64_32.ttype == mpcf.f32
    assert f64_32.vtype == mpcf.f32


"""

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

"""
