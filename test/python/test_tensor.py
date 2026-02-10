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

    with pytest.raises(IndexError) as exc:
        print(shp[3])
    


def test_empty_shape():
    shp = mpcf.TShape((2, 3, 5))

    print(f"Shpae: {shp}")
    #X = mpcf.Tensor()
    assert(False)

def test_basic_shape():
    X = mpcf.zerosT((7, 3, 5))

    print(X.shape)
    assert(False)