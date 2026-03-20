# Copyright 2024-2026 Bjorn Wehlin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

import masspcf as mpcf
from masspcf.tensor import BoolTensor, Float32Tensor


def _mpcf(arr):
    """Convert a float32 numpy array to a Float32Tensor."""
    return Float32Tensor(np.asarray(arr, dtype=np.float32))


def _assert_masked_select(arr, mask):
    """Assert that masked select matches NumPy."""
    arr = np.asarray(arr, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    result = np.asarray(_mpcf(arr)[BoolTensor(mask)])
    masked = arr[mask]
    np.testing.assert_array_equal(result, masked)
    assert result.shape == masked.shape


def _assert_masked_fill(arr, mask, fill_value):
    """Assert that masked scalar fill matches NumPy."""
    arr = np.asarray(arr, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    t = _mpcf(arr.copy())
    t[BoolTensor(mask)] = fill_value
    arr[mask] = fill_value
    np.testing.assert_array_equal(np.asarray(t), arr)
    assert t.shape == arr.shape


def _assert_masked_assign(arr, mask, values):
    """Assert that masked tensor assign matches NumPy."""
    arr = np.asarray(arr, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    values = np.asarray(values, dtype=np.float32)
    t = _mpcf(arr.copy())
    t[BoolTensor(mask)] = _mpcf(values)
    arr[mask] = values
    np.testing.assert_array_equal(np.asarray(t), arr)
    assert t.shape == arr.shape



# =============================================================================
# __getitem__ with BoolTensor mask
# =============================================================================


class TestMaskedGetitem:
    def test_basic(self):
        _assert_masked_select([0, 1, 2, 3, 4], [True, False, True, False, True])

    def test_bool_tensor(self):
        """BoolTensor[BoolTensor] should work too."""
        t = BoolTensor(np.array([True, False, True, False]))
        mask = BoolTensor(np.array([False, False, True, True]))
        np.testing.assert_array_equal(np.asarray(t[mask]), [True, False])

    def test_2d(self):
        _assert_masked_select(
            [[0, 1, 2], [3, 4, 5]],
            [[False, True, False], [True, False, True]],
        )

    def test_all_true(self):
        _assert_masked_select([10, 20, 30], [True, True, True])

    def test_all_false(self):
        _assert_masked_select([10, 20, 30], [False, False, False])

    def test_all_false_2d(self):
        _assert_masked_select(np.arange(6).reshape(2, 3), np.zeros((2, 3), dtype=bool))

    def test_shape_mismatch_raises(self):
        """Mask shape must exactly match tensor shape."""
        with pytest.raises(ValueError):
            _mpcf(np.zeros((2, 3)))[BoolTensor(np.array([True, False, True, False, True]))]

    def test_broadcast_mask_raises(self):
        """1D mask on 2D tensor raises, matching NumPy behavior."""
        with pytest.raises(ValueError):
            _mpcf(np.zeros((2, 3)))[BoolTensor(np.array([True, False, True]))]

    def test_empty(self):
        _assert_masked_select(
            np.array([], dtype=np.float32),
            np.array([], dtype=bool),
        )

    def test_pcf64(self):
        """Masked select on a Pcf64Tensor."""
        t = mpcf.zeros((3,), dtype=mpcf.pcf64)
        t[0] = mpcf.Pcf([(0, 1.0), (1, 2.0)])
        t[1] = mpcf.Pcf([(0, 3.0)])
        t[2] = mpcf.Pcf([(0, 0.0)])
        mask = BoolTensor(np.array([True, False, True]))
        assert t[mask].shape[0] == 2


# =============================================================================
# __setitem__ with BoolTensor mask — scalar fill
# =============================================================================


class TestMaskedSetitemScalar:
    def test_fill(self):
        _assert_masked_fill([0, 1, 2, 3, 4], [True, False, True, False, True], 99.0)

    def test_2d_fill(self):
        _assert_masked_fill(
            [[0, 1, 2], [3, 4, 5]],
            [[False, True, False], [True, False, True]],
            -1.0,
        )

    def test_bool_fill(self):
        np_arr = np.array([True, False, True, False])
        np_mask = np.array([True, True, False, False])
        t = BoolTensor(np_arr.copy())
        t[BoolTensor(np_mask)] = False
        np_arr[np_mask] = False
        np.testing.assert_array_equal(np.asarray(t), np_arr)

    def test_fill_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            _mpcf(np.zeros((2, 3)))[BoolTensor(np.array([True, False, True, False, True]))] = -1.0


# =============================================================================
# __setitem__ with BoolTensor mask — tensor assign
# =============================================================================


class TestMaskedSetitemTensor:
    def test_assign(self):
        _assert_masked_assign([0, 1, 2, 3, 4], [True, False, True, False, True], [10, 20, 30])

    def test_2d_assign(self):
        _assert_masked_assign(
            [[0, 1, 2], [3, 4, 5]],
            [[False, True, False], [True, True, False]],
            [10, 20, 30],
        )

    def test_values_length_mismatch_raises(self):
        t = _mpcf(np.arange(5))
        mask = BoolTensor(np.array([True, False, True, False, True]))
        with pytest.raises(ValueError):
            t[mask] = _mpcf(np.array([10, 20]))

    def test_assign_shape_mismatch_raises(self):
        t = _mpcf(np.zeros((2, 3)))
        mask = BoolTensor(np.array([True, False, True, False, True]))
        with pytest.raises(ValueError):
            t[mask] = _mpcf(np.array([10, 20, 30]))


# =============================================================================
# Non-contiguous tensors
# =============================================================================


class TestMaskedNonContiguous:
    def test_select_matches_numpy(self):
        arr = np.arange(20, dtype=np.float32).reshape(4, 5)
        np_view = arr[::2, 1:4]
        mpcf_view = _mpcf(arr)[::2, 1:4]
        mask = np.array([[True, False, True], [False, True, False]])
        np.testing.assert_array_equal(
            np.asarray(mpcf_view[BoolTensor(mask)]),
            np_view[mask],
        )

    def test_fill_matches_numpy(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        t = _mpcf(arr.copy())
        np_view = arr[::2]
        mpcf_view = t[::2]
        mask = np.array([[True, False, True, False], [False, True, False, True]])
        np_view[mask] = -1.0
        mpcf_view[BoolTensor(mask)] = -1.0
        np.testing.assert_array_equal(np.asarray(t), arr)


# =============================================================================
# Dtype validation on masked setitem
# =============================================================================


class TestMaskedSetitemDtypeValidation:
    def test_scalar_wrong_type_raises(self):
        t = _mpcf(np.array([1, 2, 3]))
        mask = BoolTensor(np.array([True, False, True]))
        with pytest.raises(TypeError):
            t[mask] = "bad"

    def test_tensor_wrong_type_raises(self):
        t = _mpcf(np.array([1, 2, 3]))
        mask = BoolTensor(np.array([True, False, True]))
        with pytest.raises(TypeError):
            t[mask] = BoolTensor(np.array([True, True]))

    def test_bool_tensor_rejects_float_scalar(self):
        t = BoolTensor(np.array([True, False, True]))
        mask = BoolTensor(np.array([True, False, False]))
        with pytest.raises(TypeError):
            t[mask] = 3.14

    def test_bool_tensor_rejects_float_tensor(self):
        t = BoolTensor(np.array([True, False, True]))
        mask = BoolTensor(np.array([True, False, False]))
        with pytest.raises(TypeError):
            t[mask] = _mpcf(np.array([99]))


# =============================================================================
# BoolTensor from numpy array
# =============================================================================


class TestBoolTensorFromNumpy:
    def test_1d(self):
        bt = BoolTensor(np.array([True, False, True]))
        np.testing.assert_array_equal(np.asarray(bt), [True, False, True])

    def test_2d(self):
        bt = BoolTensor(np.array([[True, False], [False, True]]))
        np.testing.assert_array_equal(np.asarray(bt), [[True, False], [False, True]])

    def test_from_int_array(self):
        """Integer arrays are coerced to bool."""
        bt = BoolTensor(np.array([1, 0, 1, 0]))
        np.testing.assert_array_equal(np.asarray(bt), [True, False, True, False])

    def test_empty(self):
        bt = BoolTensor(np.array([], dtype=bool))
        assert tuple(bt.shape) == (0,)