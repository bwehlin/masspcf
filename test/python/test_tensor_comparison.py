#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import numpy.testing as npt
import pytest

import masspcf as mpcf


# --- BoolTensor basics ---


def test_bool_tensor_zeros():
    bt = mpcf.zeros((3,), dtype=mpcf.boolean)
    expected = np.zeros((3,), dtype=bool)
    assert isinstance(bt, mpcf.BoolTensor)
    assert bt.shape == expected.shape
    npt.assert_array_equal(np.asarray(bt), expected)


def test_bool_tensor_bool_single_element():
    np_a = np.array([1.0])
    np_b = np.array([1.0])
    np_c = np.array([2.0])

    a = mpcf.Float64Tensor(np_a)
    b = mpcf.Float64Tensor(np_b)
    c = mpcf.Float64Tensor(np_c)

    assert a.shape == np_a.shape

    assert bool(a == b) is bool(np_a == np_b)
    assert bool(a == c) is bool(np_a == np_c)


def test_bool_tensor_bool_multi_element_raises():
    np_a = np.array([1.0, 2.0])
    np_b = np.array([1.0, 2.0])

    # NumPy raises on bool() of multi-element array; mpcf should too
    with pytest.raises(ValueError):
        bool(np_a == np_b)

    a = mpcf.Float64Tensor(np_a)
    b = mpcf.Float64Tensor(np_b)
    with pytest.raises(ValueError, match="more than one element"):
        bool(a == b)


# --- Elementwise comparisons ---


def test_eq_elementwise():
    np_a = np.array([1.0, 2.0, 3.0])
    np_b = np.array([1.0, 9.0, 3.0])
    result = mpcf.Float64Tensor(np_a) == mpcf.Float64Tensor(np_b)
    assert isinstance(result, mpcf.BoolTensor)
    npt.assert_array_equal(np.asarray(result), np_a == np_b)


def test_ne_elementwise():
    np_a = np.array([1.0, 2.0, 3.0])
    np_b = np.array([1.0, 9.0, 3.0])
    result = mpcf.Float64Tensor(np_a) != mpcf.Float64Tensor(np_b)
    assert isinstance(result, mpcf.BoolTensor)
    npt.assert_array_equal(np.asarray(result), np_a != np_b)


def test_lt_elementwise():
    np_a = np.array([1.0, 5.0, 3.0])
    np_b = np.array([2.0, 4.0, 3.0])
    result = mpcf.Float64Tensor(np_a) < mpcf.Float64Tensor(np_b)
    assert isinstance(result, mpcf.BoolTensor)
    npt.assert_array_equal(np.asarray(result), np_a < np_b)


def test_le_elementwise():
    np_a = np.array([1.0, 5.0, 3.0])
    np_b = np.array([2.0, 4.0, 3.0])
    result = mpcf.Float64Tensor(np_a) <= mpcf.Float64Tensor(np_b)
    assert isinstance(result, mpcf.BoolTensor)
    npt.assert_array_equal(np.asarray(result), np_a <= np_b)


def test_gt_elementwise():
    np_a = np.array([1.0, 5.0, 3.0])
    np_b = np.array([2.0, 4.0, 3.0])
    result = mpcf.Float64Tensor(np_a) > mpcf.Float64Tensor(np_b)
    assert isinstance(result, mpcf.BoolTensor)
    npt.assert_array_equal(np.asarray(result), np_a > np_b)


def test_ge_elementwise():
    np_a = np.array([1.0, 5.0, 3.0])
    np_b = np.array([2.0, 4.0, 3.0])
    result = mpcf.Float64Tensor(np_a) >= mpcf.Float64Tensor(np_b)
    assert isinstance(result, mpcf.BoolTensor)
    npt.assert_array_equal(np.asarray(result), np_a >= np_b)


# --- Broadcasting comparisons ---


def test_eq_broadcast_row():
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    np_b = np.array([1.0, 4.0])
    expected = np_a == np_b
    result = mpcf.Float64Tensor(np_a) == mpcf.Float64Tensor(np_b)
    assert isinstance(result, mpcf.BoolTensor)
    assert result.shape == expected.shape
    npt.assert_array_equal(np.asarray(result), expected)


def test_lt_broadcast_col():
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    np_b = np.array([[2.0], [3.0]])
    expected = np_a < np_b
    result = mpcf.Float64Tensor(np_a) < mpcf.Float64Tensor(np_b)
    assert isinstance(result, mpcf.BoolTensor)
    assert result.shape == expected.shape
    npt.assert_array_equal(np.asarray(result), expected)


def test_ge_broadcast_scalar_tensor():
    np_a = np.array([1.0, 2.0, 3.0])
    np_b = np.array([2.0])
    expected = np_a >= np_b
    result = mpcf.Float64Tensor(np_a) >= mpcf.Float64Tensor(np_b)
    assert isinstance(result, mpcf.BoolTensor)
    assert result.shape == expected.shape
    npt.assert_array_equal(np.asarray(result), expected)


# --- array_equal ---


def test_array_equal_true():
    np_a = np.array([1.0, 2.0, 3.0])
    a = mpcf.Float64Tensor(np_a)
    assert a.array_equal(a) == np.array_equal(np_a, np_a)
    assert a.array_equal(a.copy()) == np.array_equal(np_a, np_a.copy())


def test_array_equal_false():
    np_a = np.array([1.0, 2.0, 3.0])
    np_b = np.array([1.0, 9.0, 3.0])
    a = mpcf.Float64Tensor(np_a)
    b = mpcf.Float64Tensor(np_b)
    assert a.array_equal(b) == np.array_equal(np_a, np_b)


def test_array_equal_different_shape():
    np_a = np.array([1.0, 2.0])
    np_b = np.array([1.0, 2.0, 3.0])
    a = mpcf.Float64Tensor(np_a)
    b = mpcf.Float64Tensor(np_b)
    assert a.array_equal(b) == np.array_equal(np_a, np_b)


def test_array_equal_numpy():
    np_a = np.array([1.0, 2.0, 3.0])
    np_b = np.array([1.0, 9.0, 3.0])
    a = mpcf.Float64Tensor(np_a)
    assert a.array_equal(np_a) == np.array_equal(np_a, np_a)
    assert a.array_equal(np_b) == np.array_equal(np_a, np_b)


# --- PCF tensor elementwise eq ---


def test_pcf_tensor_eq():
    a = mpcf.random.noisy_sin((3,))
    b = a.copy()
    result = a == b
    assert isinstance(result, mpcf.BoolTensor)
    npt.assert_array_equal(np.asarray(result), [True, True, True])


def test_pcf_tensor_array_equal():
    a = mpcf.random.noisy_sin((3,))
    b = a.copy()
    assert a.array_equal(b)
