import pytest
import masspcf as mpcf

import numpy as np

from utils import np_strides_in_items, print_np_array_details

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
    
    Xnp = np.zeros((6))
    Ynp = Xnp[0:5:2]

    assert Y.shape == Ynp.shape
    assert Y[0] == 0
    assert Y[1] == 2
    assert Y[2] == 4

def test_extract_with_step():
    X = mpcf.zerosT((3, 9, 2))

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                X[i, j, k] = 100 * i + 10 * j + k
    
    Y = X[:, 0:7:2, :] # start:stop:step

    Xnp = np.zeros((3, 9, 2))
    Ynp = Xnp[:, 0:7:2, :]

    assert Y.shape == Ynp.shape # (3, 4 ,2)
    assert Y.strides == np_strides_in_items(Ynp)

    print(Y.shape)


    assert Y[0, 0, 0] == X[0, 0, 0]
    assert Y[0, 0, 1] == X[0, 0, 1]

    assert Y[0, 1, 0] == X[0, 2, 0]
    assert Y[0, 1, 1] == X[0, 2, 1]

    assert Y[0, 2, 0] == X[0, 4, 0]
    assert Y[0, 2, 1] == X[0, 4, 1]

    assert Y[0, 3, 0] == X[0, 6, 0]
    assert Y[0, 3, 1] == X[0, 6, 1]

    assert Y[1, 0, 0] == X[1, 0, 0]
    assert Y[1, 0, 1] == X[1, 0, 1]

    assert Y[1, 1, 0] == X[1, 2, 0]
    assert Y[1, 1, 1] == X[1, 2, 1]

    assert Y[1, 2, 0] == X[1, 4, 0]
    assert Y[1, 2, 1] == X[1, 4, 1]

    assert Y[1, 3, 0] == X[1, 6, 0]
    assert Y[1, 3, 1] == X[1, 6, 1]

    assert Y[2, 0, 0] == X[2, 0, 0]
    assert Y[2, 0, 1] == X[2, 0, 1]

    assert Y[2, 1, 0] == X[2, 2, 0]
    assert Y[2, 1, 1] == X[2, 2, 1]

    assert Y[2, 2, 0] == X[2, 4, 0]
    assert Y[2, 2, 1] == X[2, 4, 1]

    assert Y[2, 3, 0] == X[2, 6, 0]
    assert Y[2, 3, 1] == X[2, 6, 1]

def test_extract_with_offsets():
    X = mpcf.zerosT((7, 9, 5))

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                X[i, j, k] = 100 * i + 10 * j + k
        
    Y = X[1::3, 3:7:2, 2:5] # start:stop:step

    assert Y.shape == mpcf.TShape((2, 2, 3))
    assert Y.strides == [135, 10, 1]
    assert Y.offset == 62

    assert Y[1, 1, 0] == X[4, 5, 2]

def test_recursive_extract():
    X = mpcf.zerosT((9, 8, 7, 6))
    Xnp = np.zeros((9, 8, 7, 6))

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                for l in range(X.shape[3]):
                    X[i, j, k, l] = 1000 * i + 100 * j + 10 * k + l
                    Xnp[i, j, k, l] = 1000 * i + 100 * j + 10 * k + l
    
    Y0 = X[4:9:2, ::3, 2:5, 1:3:2]
    Y0np = Xnp[4:9:2, ::3, 2:5, 1:3:2]

    assert Y0.shape == Y0np.shape
    assert Y0.strides == np_strides_in_items(Y0np)

    for i in range(Y0.shape[0]):
        for j in range(Y0.shape[1]):
            for k in range(Y0.shape[2]):
                for l in range(Y0.shape[3]):
                    assert Y0[i,j,k,l] == Y0np[i,j,k,l]

    Y1 = Y0[1:3, :, :2, 0]
    Y1np = Y0np[1:3, :, :2, 0]

    assert Y1.shape == Y1np.shape

    assert Y1.strides == np_strides_in_items(Y1np)

    assert Y1[0, 0, 0] == 6021
    assert Y1[0, 0, 1] == 6031

    assert Y1[0, 1, 0] == 6321
    assert Y1[0, 1, 1] == 6331

    assert Y1[0, 2, 0] == 6621
    assert Y1[0, 2, 1] == 6631

    assert Y1[1, 0, 0] == 8021
    assert Y1[1, 0, 1] == 8031

    assert Y1[1, 1, 0] == 8321
    assert Y1[1, 1, 1] == 8331

    assert Y1[1, 2, 0] == 8621
    assert Y1[1, 2, 1] == 8631





