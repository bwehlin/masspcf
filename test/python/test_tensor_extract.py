import pytest
import masspcf as mpcf

import numpy as np

def test_extract_element():
    X = mpcf.zerosT((2, 3))

    print(X.strides)

    assert X[1, 0] == 0.0

    X[0, 1] = 2.0

    assert X[0, 1] == 2.0

def test_extract_element1d():
    X = mpcf.zerosT((2))
    X[1] = 2.0

    assert X[0] == 0.0
    assert X[1] == 2.0

def test_extract_subtensor():
    X = mpcf.zerosT((3, 4, 5))

    for i in range(3):
        for j in range(4):
            for k in range(5):
                X[i, j, k] = 100 * i + 10 * j + k

    Y = X[:, 1:3, 2:4]

    assert Y.shape == mpcf.TShape((3, 2, 2))

    assert Y[0, 0, 0] == X[0, 1, 2]
    assert Y[0, 0, 1] == X[0, 1, 3]

    assert Y[0, 1, 0] == X[0, 2, 2]
    assert Y[0, 1, 1] == X[0, 2, 3]

    assert Y[1, 0, 0] == X[1, 1, 2]
    assert Y[1, 0, 1] == X[1, 1, 3]

    assert Y[1, 1, 0] == X[1, 2, 2]
    assert Y[1, 1, 1] == X[1, 2, 3]

    assert Y[2, 0, 0] == X[2, 1, 2]
    assert Y[2, 0, 1] == X[2, 1, 3]

    assert Y[2, 1, 0] == X[2, 2, 2]
    assert Y[2, 1, 1] == X[2, 2, 3]

    
def test_extract1d_with_step():
    X = mpcf.zerosT((6))

    for i in range(X.shape[0]):
        X[i] = i
    
    Y = X[0:5:2] # start:stop:step
    
    assert Y.shape == mpcf.TShape((2))
    assert Y[0] == 0
    assert Y[1] == 2

def test_extract_with_step():
    X = mpcf.zerosT((3, 9, 2))

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                X[i, j, k] = 100 * i + 10 * j + k
    
    Y = X[:, 0:7:2, :] # start:stop:step

    assert Y.shape == mpcf.TShape((3, 3, 2))

    assert Y[0, 0, 0] == X[0, 0, 0]
    assert Y[0, 0, 1] == X[0, 0, 1]

    assert Y[0, 1, 0] == X[0, 2, 0]
    assert Y[0, 1, 1] == X[0, 2, 1]

    assert Y[0, 2, 0] == X[0, 4, 0]
    assert Y[0, 2, 1] == X[0, 4, 1]

    assert Y[1, 0, 0] == X[1, 0, 0]
    assert Y[1, 0, 1] == X[1, 0, 1]

    assert Y[1, 1, 0] == X[1, 2, 0]
    assert Y[1, 1, 1] == X[1, 2, 1]

    assert Y[1, 2, 0] == X[1, 4, 0]
    assert Y[1, 2, 1] == X[1, 4, 1]

    assert Y[2, 0, 0] == X[2, 0, 0]
    assert Y[2, 0, 1] == X[2, 0, 1]

    assert Y[2, 1, 0] == X[2, 2, 0]
    assert Y[2, 1, 1] == X[2, 2, 1]

    assert Y[2, 2, 0] == X[2, 4, 0]
    assert Y[2, 2, 1] == X[2, 4, 1]

    
    