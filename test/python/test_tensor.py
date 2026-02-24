import pytest
import masspcf as mpcf
import masspcf._mpcf_cpp as mcpp

import numpy as np

def test_shape_dunder():
    shp = mpcf.Shape((2, 3, 5))

    str_ = str(shp)
    repr_ = repr(shp)

    assert(str_ == "Shape(2, 3, 5)")
    assert(repr_ == "(2, 3, 5)")

    assert(len(shp) == 3)
    assert(shp[0] == 2)
    assert(shp[1] == 3)
    assert(shp[2] == 5)

    with pytest.raises(IndexError):
        print(shp[3])
    
def test_basic_shape_strides():
    X = mpcf.zeros((7, 3, 5, 6), dtype=mpcf.f64)

    assert X.shape == mpcf.Shape((7, 3, 5, 6))
    assert X.shape != mpcf.Shape((1, 2, 3))

    assert X.strides[0] == 3 * 5 * 6 * 1    # 90
    assert X.strides[1] == 5 * 6 * 1        # 30
    assert X.strides[2] == 6 * 1            #  6
    assert X.strides[3] == 1                #  1

def test_construct_1d_tensor():
    s = mpcf.Shape((10))
    X = mpcf.zeros(s)
    
    assert X.shape == s

def test_dtype_results_in_correct_type():
    X32 = mpcf.zeros((3, 2), dtype=mpcf.pcf32)
    assert isinstance(X32, mpcf.Pcf32Tensor)

    assert isinstance(X32[0, 0], mpcf.Pcf)
    assert isinstance(X32[0, 0]._data, mcpp.Pcf_f32_f32)

    X64 = mpcf.zeros((3, 2), dtype=mpcf.pcf64)
    assert isinstance(X64, mpcf.Pcf64Tensor)

    assert isinstance(X64[0, 0], mpcf.Pcf)
    assert isinstance(X64[0, 0]._data, mcpp.Pcf_f64_f64)

def test_tensor_copy_does_not_modify_original():
    X = mpcf.zeros((10, 10), dtype=mpcf.f64)

    X[2, 3] = 1.5

    Y = X.copy()

    Y[2, 3] = 0.5

    assert Y[2, 3] != X[2, 3]

    X[2, 3] = 4.5

    assert Y[2, 3] != X[2, 3]

