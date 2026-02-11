import pytest
import masspcf as mpcf

import numpy as np

def test_shape_dunder():
    shp = mpcf.TShape((2, 3, 5))

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
    X = mpcf.zerosT((7, 3, 5, 6))

    assert X.shape == mpcf.TShape((7, 3, 5, 6))
    assert X.shape != mpcf.TShape((1, 2, 3))

    assert X.strides[0] == 3 * 5 * 6 * 1    # 90
    assert X.strides[1] == 5 * 6 * 1        # 30
    assert X.strides[2] == 6 * 1            #  6
    assert X.strides[3] == 1                #  1

def test_construct_1d_tensor():
    s = mpcf.TShape((10))
    X = mpcf.zerosT(s)
    
    assert X.shape == s