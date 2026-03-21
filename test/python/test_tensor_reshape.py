import numpy as np
import pytest

import masspcf as mpcf


_NUMERIC_TYPES = [
    pytest.param(mpcf.FloatTensor, np.float64, id="float64"),
    pytest.param(mpcf.FloatTensor, np.float32, id="float32"),
    pytest.param(mpcf.IntTensor, np.int32, id="int32"),
    pytest.param(mpcf.IntTensor, np.int64, id="int64"),
]


def _assert_reshape(np_arr, new_shape, TensorType):
    """Assert that mpcf reshape matches NumPy."""
    t = TensorType(np_arr)
    result = np.asarray(t.reshape(new_shape))
    expected = np_arr.reshape(new_shape)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == expected.shape


def test_tensor2d_flatten():
    np_arr = np.array([[0, 1, 2], [10, 11, 12]], dtype=np.float32)
    t = mpcf.FloatTensor(np_arr)
    np_flat = np_arr.flatten()
    flat = t.flatten()
    assert flat.shape == np_flat.shape
    for i in range(flat.shape[0]):
        assert flat[i] == np_flat[i]


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestReshape:
    def test_2d_to_1d(self, TensorType, np_dtype):
        _assert_reshape(np.arange(12, dtype=np_dtype).reshape(3, 4), (12,), TensorType)

    def test_1d_to_2d(self, TensorType, np_dtype):
        _assert_reshape(np.arange(12, dtype=np_dtype), (3, 4), TensorType)

    def test_2d_to_3d(self, TensorType, np_dtype):
        _assert_reshape(np.arange(24, dtype=np_dtype).reshape(4, 6), (2, 3, 4), TensorType)

    def test_infer_dimension(self, TensorType, np_dtype):
        _assert_reshape(np.arange(24, dtype=np_dtype), (6, -1), TensorType)

    def test_infer_first_dimension(self, TensorType, np_dtype):
        _assert_reshape(np.arange(24, dtype=np_dtype), (-1, 4), TensorType)

    def test_same_shape(self, TensorType, np_dtype):
        _assert_reshape(np.arange(12, dtype=np_dtype).reshape(3, 4), (3, 4), TensorType)

    def test_non_contiguous(self, TensorType, np_dtype):
        np_arr = np.arange(12, dtype=np_dtype).reshape(3, 4)
        t = TensorType(np_arr)
        view = t[::2, :]  # non-contiguous
        _assert_reshape(np.asarray(view), (8,), TensorType)

    def test_contiguous_is_view(self, TensorType, np_dtype):
        np_arr = np.arange(12, dtype=np_dtype)
        t = TensorType(np_arr)
        reshaped = t.reshape((3, 4))
        reshaped[0, 0] = 99
        assert t[0] == 99

    def test_non_contiguous_is_copy(self, TensorType, np_dtype):
        np_arr = np.arange(12, dtype=np_dtype).reshape(3, 4)
        t = TensorType(np_arr)
        view = t[::2, :]
        reshaped = view.reshape((8,))
        reshaped[0] = 99
        assert t[0, 0] == 0

    def test_incompatible_shape_raises(self, TensorType, np_dtype):
        t = TensorType(np.arange(12, dtype=np_dtype))
        with pytest.raises(ValueError):
            t.reshape((5, 5))

    def test_multiple_infer_raises(self, TensorType, np_dtype):
        t = TensorType(np.arange(12, dtype=np_dtype))
        with pytest.raises(ValueError):
            t.reshape((-1, -1))
