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


def test_extract_element():
    X = mpcf.zerosT((2, 3))

    print(X.strides)

    assert X[1, 0] == 0.0

    X[0, 1] = 2.0

    assert X[0, 1] == 2.0

def test_extract_submatrix():
    X = mpcf.zerosT((3, 4))

    for i in range(3):
        for j in range(4):
            X[i, j] = 10 * i + j

    Y = X[:, 1:3]

    assert Y.shape == mpcf.TShape((3, 2))

    assert Y[0, 0] == X[0, 1]
    assert Y[0, 1] == X[0, 2]

    assert Y[1, 0] == X[1, 1]
    assert Y[1, 1] == X[1, 2]

    assert Y[2, 0] == X[2, 1]
    assert Y[2, 1] == X[2, 2]
