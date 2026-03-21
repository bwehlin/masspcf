import numpy as np

import masspcf as mpcf
from masspcf.tensor import FloatTensor


def test_tensor2d_flatten():
    np_arr = np.array([[0, 1, 2], [10, 11, 12]], dtype=np.float32)
    t = FloatTensor(np_arr)
    np_flat = np_arr.flatten()

    flat = t.flatten()

    assert flat.shape == np_flat.shape

    for i in range(flat.shape[0]):
        assert flat[i] == np_flat[i]
