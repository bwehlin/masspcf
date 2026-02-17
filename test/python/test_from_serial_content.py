import masspcf as mpcf
import numpy as np
import pytest

def test_fsc_shapes():
    content = np.zeros((0, 2))
    enumeration = np.zeros((0, 2))

    X = mpcf.from_serial_content(content, enumeration)

    assert len(X.shape) == 1
    assert X.shape[0] == 0

    content = np.zeros((1000, 2))
    enumeration = np.zeros((10, 20, 30, 2), dtype=np.longlong)
    enumeration[:,:,:, 1] = 1 # To make every stop > start

    X = mpcf.from_serial_content(content, enumeration)

    assert len(X.shape) == 3
    assert X.shape[0] == 10
    assert X.shape[1] == 20
    assert X.shape[2] == 30

def test_fsc_single():
    content = np.array([[0., 10.], [10., 20.], [20., 30.]])
    enumeration = np.array([[0, 3]])

    X = mpcf.from_serial_content(content, enumeration)

    assert X[0] == mpcf.Pcf(content[0:3])

def test_fsc_multiple():
    content = np.array([[0., 10.], [10., 20.], [20., 30.], [0., 50.], [10., 60.]])
    enumeration = np.array([[0, 3], [3, 5]])

    X = mpcf.from_serial_content(content, enumeration)

    assert X[0] == mpcf.Pcf(content[0:3])
    assert X[1] == mpcf.Pcf(content[3:5])

def test_fsc_multidim():
    content = np.array([[0., 10.], [10., 20.], [20., 30.],         [0., 50.], [10., 60.],
                        [0., 5.], [0., 6.], [0., 7.], [0., 8.],    [0., 0.],
                        [0., 2.], [1., 3.], [3., 0.],              [0., 10.], [0., 100.]])

    enumeration = np.array([[[0, 3], [3, 5]],
                            [[5, 9], [9, 10]],
                            [[10, 13], [13, 15]] ])


    X = mpcf.from_serial_content(content, enumeration)

    assert X.shape == (3, 2)

    assert X[0, 0] == mpcf.Pcf(content[enumeration[0, 0, 0]:enumeration[0, 0, 1]])
    assert X[0, 1] == mpcf.Pcf(content[enumeration[0, 1, 0]:enumeration[0, 1, 1]])
    assert X[1, 0] == mpcf.Pcf(content[enumeration[1, 0, 0]:enumeration[1, 0, 1]])
    assert X[1, 1] == mpcf.Pcf(content[enumeration[1, 1, 0]:enumeration[1, 1, 1]])
    assert X[2, 0] == mpcf.Pcf(content[enumeration[2, 0, 0]:enumeration[2, 0, 1]])
    assert X[2, 1] == mpcf.Pcf(content[enumeration[2, 1, 0]:enumeration[2, 1, 1]])

def test_fsc_stop_le_start_throws():
    content = np.array([[0., 10.], [10., 20.], [20., 30.], [0., 50.], [10., 60.]])
    enumeration = np.array([[3, 0], [3, 5]])

    with pytest.raises(ValueError):
        X = mpcf.from_serial_content(content, enumeration)

def test_fsc_dtypes():
    content32 = np.array([[0., 10.], [10., 20.], [20., 30.], [0., 50.], [10., 60.]], dtype=np.float32)
    content64 = np.array([[0., 10.], [10., 20.], [20., 30.], [0., 50.], [10., 60.]], dtype=np.float64)
    enumeration = np.array([[0, 3], [3, 5]])

    X32 = mpcf.from_serial_content(content32, enumeration)
    assert X32.dtype == mpcf.float32

    X64 = mpcf.from_serial_content(content64, enumeration)
    assert X64.dtype == mpcf.float64

    X32_64 = mpcf.from_serial_content(content32, enumeration, dtype=mpcf.float64)
    assert X32_64.dtype == mpcf.float64

    X64_32 = mpcf.from_serial_content(content32, enumeration, dtype=mpcf.float32)
    assert X64_32.dtype == mpcf.float32
