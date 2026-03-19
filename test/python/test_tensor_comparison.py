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
    assert isinstance(bt, mpcf.BoolTensor)
    assert bt.shape == (3,)
    npt.assert_array_equal(np.asarray(bt), [False, False, False])


def test_bool_tensor_bool_single_element():
    a = mpcf.Float64Tensor(np.array([1.0]))
    b = mpcf.Float64Tensor(np.array([1.0]))
    assert bool(a == b) is True

    c = mpcf.Float64Tensor(np.array([2.0]))
    assert bool(a == c) is False


def test_bool_tensor_bool_multi_element_raises():
    a = mpcf.Float64Tensor(np.array([1.0, 2.0]))
    b = mpcf.Float64Tensor(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="more than one element"):
        bool(a == b)


# --- Elementwise == and != ---


def _check_cmp(np_a, np_b, op_str):
    """Compare numpy and mpcf results for a comparison operator."""
    ops = {
        "==": (lambda a, b: a == b, np.equal),
        "!=": (lambda a, b: a != b, np.not_equal),
        "<": (lambda a, b: a < b, np.less),
        "<=": (lambda a, b: a <= b, np.less_equal),
        ">": (lambda a, b: a > b, np.greater),
        ">=": (lambda a, b: a >= b, np.greater_equal),
    }
    mpcf_op, np_op = ops[op_str]

    mpcf_result = mpcf_op(mpcf.Float64Tensor(np_a), mpcf.Float64Tensor(np_b))
    np_result = np_op(np_a, np_b)

    assert isinstance(mpcf_result, mpcf.BoolTensor)
    npt.assert_array_equal(np.asarray(mpcf_result), np_result)


def test_eq_elementwise():
    _check_cmp(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 9.0, 3.0]),
        "==",
    )


def test_ne_elementwise():
    _check_cmp(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 9.0, 3.0]),
        "!=",
    )


def test_lt_elementwise():
    _check_cmp(
        np.array([1.0, 5.0, 3.0]),
        np.array([2.0, 4.0, 3.0]),
        "<",
    )


def test_le_elementwise():
    _check_cmp(
        np.array([1.0, 5.0, 3.0]),
        np.array([2.0, 4.0, 3.0]),
        "<=",
    )


def test_gt_elementwise():
    _check_cmp(
        np.array([1.0, 5.0, 3.0]),
        np.array([2.0, 4.0, 3.0]),
        ">",
    )


def test_ge_elementwise():
    _check_cmp(
        np.array([1.0, 5.0, 3.0]),
        np.array([2.0, 4.0, 3.0]),
        ">=",
    )


# --- Broadcasting comparisons (matched against NumPy) ---


def test_eq_broadcast_row():
    _check_cmp(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([1.0, 4.0]),
        "==",
    )


def test_lt_broadcast_col():
    _check_cmp(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[2.0], [3.0]]),
        "<",
    )


def test_ge_broadcast_scalar_tensor():
    _check_cmp(
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0]),
        ">=",
    )


# --- array_equal ---


def test_array_equal_true():
    a = mpcf.Float64Tensor(np.array([1.0, 2.0, 3.0]))
    assert a.array_equal(a)
    assert a.array_equal(a.copy())


def test_array_equal_false():
    a = mpcf.Float64Tensor(np.array([1.0, 2.0, 3.0]))
    b = mpcf.Float64Tensor(np.array([1.0, 9.0, 3.0]))
    assert not a.array_equal(b)


def test_array_equal_different_shape():
    a = mpcf.Float64Tensor(np.array([1.0, 2.0]))
    b = mpcf.Float64Tensor(np.array([1.0, 2.0, 3.0]))
    assert not a.array_equal(b)


def test_array_equal_numpy():
    a = mpcf.Float64Tensor(np.array([1.0, 2.0, 3.0]))
    assert a.array_equal(np.array([1.0, 2.0, 3.0]))
    assert not a.array_equal(np.array([1.0, 9.0, 3.0]))


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
