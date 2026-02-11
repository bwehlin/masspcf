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
    X = mpcf.zerosT((7, 3, 5))

    assert X.shape == mpcf.TShape((7, 3, 5))
    assert X.shape != mpcf.TShape((1, 2, 3, 4))

    print(X.strides)

    assert X.strides[0] == 7 * 3 * 1
    assert X.strides[1] == 7 * 1
    assert X.strides[2] == 1

def test_construct_1d_tensor():
    s = mpcf.TShape((10))
    X = mpcf.zerosT(s)
    
    assert X.shape == s