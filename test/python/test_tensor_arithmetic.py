import numpy as np
import numpy.testing as npt
import pytest

import masspcf as mpcf


def _make_pcf32(vals):
    return mpcf.Pcf(np.array(vals, dtype=np.float32))


def _make_pcf64(vals):
    return mpcf.Pcf(np.array(vals, dtype=np.float64))


# --- Float tensors ---


class TestFloat64TensorArithmetic:
    def test_add(self):
        X = mpcf.Float64Tensor(np.array([1.0, 2.0, 3.0]))
        Y = X + 10.0
        npt.assert_array_equal(np.asarray(Y), [11.0, 12.0, 13.0])

    def test_iadd(self):
        X = mpcf.Float64Tensor(np.array([1.0, 2.0, 3.0]))
        X += 10.0
        npt.assert_array_equal(np.asarray(X), [11.0, 12.0, 13.0])

    def test_sub(self):
        X = mpcf.Float64Tensor(np.array([10.0, 20.0, 30.0]))
        Y = X - 1.0
        npt.assert_array_equal(np.asarray(Y), [9.0, 19.0, 29.0])

    def test_isub(self):
        X = mpcf.Float64Tensor(np.array([10.0, 20.0, 30.0]))
        X -= 1.0
        npt.assert_array_equal(np.asarray(X), [9.0, 19.0, 29.0])

    def test_mul(self):
        X = mpcf.Float64Tensor(np.array([1.0, 2.0, 3.0]))
        Y = X * 5.0
        npt.assert_array_equal(np.asarray(Y), [5.0, 10.0, 15.0])

    def test_imul(self):
        X = mpcf.Float64Tensor(np.array([1.0, 2.0, 3.0]))
        X *= 5.0
        npt.assert_array_equal(np.asarray(X), [5.0, 10.0, 15.0])

    def test_truediv(self):
        X = mpcf.Float64Tensor(np.array([10.0, 20.0, 30.0]))
        Y = X / 2.0
        npt.assert_array_equal(np.asarray(Y), [5.0, 10.0, 15.0])

    def test_itruediv(self):
        X = mpcf.Float64Tensor(np.array([10.0, 20.0, 30.0]))
        X /= 2.0
        npt.assert_array_equal(np.asarray(X), [5.0, 10.0, 15.0])

    def test_radd(self):
        X = mpcf.Float64Tensor(np.array([1.0, 2.0, 3.0]))
        Y = 10.0 + X
        npt.assert_array_equal(np.asarray(Y), [11.0, 12.0, 13.0])

    def test_rsub(self):
        X = mpcf.Float64Tensor(np.array([1.0, 2.0, 3.0]))
        Y = 10.0 - X
        npt.assert_array_equal(np.asarray(Y), [9.0, 8.0, 7.0])

    def test_rmul(self):
        X = mpcf.Float64Tensor(np.array([1.0, 2.0, 3.0]))
        Y = 5.0 * X
        npt.assert_array_equal(np.asarray(Y), [5.0, 10.0, 15.0])

    def test_add_does_not_modify_original(self):
        X = mpcf.Float64Tensor(np.array([1.0, 2.0, 3.0]))
        _ = X + 10.0
        npt.assert_array_equal(np.asarray(X), [1.0, 2.0, 3.0])

    def test_2d(self):
        X = mpcf.Float64Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        Y = X * 2.0
        npt.assert_array_equal(np.asarray(Y), [[2.0, 4.0], [6.0, 8.0]])


class TestFloat32TensorArithmetic:
    def test_add(self):
        X = mpcf.Float32Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        Y = X + 10.0
        npt.assert_array_almost_equal(np.asarray(Y), [11.0, 12.0, 13.0])

    def test_mul(self):
        X = mpcf.Float32Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        Y = X * 5.0
        npt.assert_array_almost_equal(np.asarray(Y), [5.0, 10.0, 15.0])

    def test_truediv(self):
        X = mpcf.Float32Tensor(np.array([10.0, 20.0, 30.0], dtype=np.float32))
        Y = X / 2.0
        npt.assert_array_almost_equal(np.asarray(Y), [5.0, 10.0, 15.0])


# --- PCF tensors (parametrized across all PCF dtypes) ---


_PCF_DTYPES = [
    pytest.param(mpcf.pcf32, np.float32, mpcf.Pcf32Tensor, id="pcf32"),
    pytest.param(mpcf.pcf64, np.float64, mpcf.Pcf64Tensor, id="pcf64"),
    pytest.param(mpcf.pcf32i, np.int32, mpcf.Pcf32iTensor, id="pcf32i"),
    pytest.param(mpcf.pcf64i, np.int64, mpcf.Pcf64iTensor, id="pcf64i"),
]


def _make_pcf(np_dtype, vals):
    return mpcf.Pcf(np.array(vals, dtype=np_dtype))


@pytest.mark.parametrize("pcf_dtype, np_dtype, tensor_cls", _PCF_DTYPES)
class TestPcfTensorArithmetic:
    def _make_tensor(self, pcf_dtype, np_dtype):
        X = mpcf.zeros((2,), dtype=pcf_dtype)
        X[0] = _make_pcf(np_dtype, [[0, 1], [2, 3]])
        X[1] = _make_pcf(np_dtype, [[0, 10], [2, 20]])
        return X

    def test_add_pcf(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        pcf = _make_pcf(np_dtype, [[0, 100], [2, 200]])
        Y = X + pcf
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(101)
        assert Y[1].to_numpy()[0, 1] == np_dtype(110)

    def test_iadd_pcf(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        pcf = _make_pcf(np_dtype, [[0, 100], [2, 200]])
        X += pcf
        assert X[0].to_numpy()[0, 1] == np_dtype(101)

    def test_sub_pcf(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        pcf = _make_pcf(np_dtype, [[0, 1], [2, 1]])
        Y = X - pcf
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(0)
        assert Y[1].to_numpy()[0, 1] == np_dtype(9)

    def test_isub_pcf(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        pcf = _make_pcf(np_dtype, [[0, 1], [2, 1]])
        X -= pcf
        assert X[0].to_numpy()[0, 1] == np_dtype(0)

    def test_add_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        Y = X + np_dtype(5)
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(6)
        assert Y[1].to_numpy()[0, 1] == np_dtype(15)

    def test_iadd_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        X += np_dtype(5)
        assert X[0].to_numpy()[0, 1] == np_dtype(6)

    def test_radd_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        Y = np_dtype(5) + X
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(6)

    def test_sub_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        Y = X - np_dtype(1)
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(0)
        assert Y[1].to_numpy()[0, 1] == np_dtype(9)

    def test_rsub_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        Y = np_dtype(100) - X
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(99)
        assert Y[1].to_numpy()[0, 1] == np_dtype(90)

    def test_isub_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        X -= np_dtype(1)
        assert X[0].to_numpy()[0, 1] == np_dtype(0)

    def test_mul_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        Y = X * np_dtype(2)
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(2)
        assert Y[1].to_numpy()[0, 1] == np_dtype(20)

    def test_imul_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        X *= np_dtype(2)
        assert X[0].to_numpy()[0, 1] == np_dtype(2)

    def test_truediv_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = mpcf.zeros((2,), dtype=pcf_dtype)
        X[0] = _make_pcf(np_dtype, [[0, 4], [2, 8]])
        X[1] = _make_pcf(np_dtype, [[0, 10], [2, 20]])
        Y = X / np_dtype(2)
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(2)
        assert Y[1].to_numpy()[0, 1] == np_dtype(5)

    def test_rtruediv_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = mpcf.zeros((2,), dtype=pcf_dtype)
        X[0] = _make_pcf(np_dtype, [[0, 2], [2, 4]])
        X[1] = _make_pcf(np_dtype, [[0, 5], [2, 10]])
        Y = np_dtype(20) / X
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(10)
        assert Y[1].to_numpy()[0, 1] == np_dtype(4)

    def test_itruediv_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = mpcf.zeros((2,), dtype=pcf_dtype)
        X[0] = _make_pcf(np_dtype, [[0, 4], [2, 8]])
        X[1] = _make_pcf(np_dtype, [[0, 10], [2, 20]])
        X /= np_dtype(2)
        assert X[0].to_numpy()[0, 1] == np_dtype(2)

    def test_add_does_not_modify_original(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        orig = X[0].to_numpy().copy()
        _ = X + _make_pcf(np_dtype, [[0, 100], [2, 200]])
        npt.assert_array_equal(X[0].to_numpy(), orig)

    def test_mul_does_not_modify_original(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        orig = X[0].to_numpy().copy()
        _ = X * np_dtype(10)
        npt.assert_array_equal(X[0].to_numpy(), orig)


@pytest.mark.parametrize("pcf_dtype, np_dtype, tensor_cls", _PCF_DTYPES)
class TestPcfTensorMulDiv:
    def test_mul_tensor_tensor(self, pcf_dtype, np_dtype, tensor_cls):
        X = mpcf.zeros((2,), dtype=pcf_dtype)
        X[0] = _make_pcf(np_dtype, [[0, 2], [2, 4]])
        X[1] = _make_pcf(np_dtype, [[0, 10], [2, 20]])
        Y = mpcf.zeros((2,), dtype=pcf_dtype)
        Y[0] = _make_pcf(np_dtype, [[0, 3], [2, 5]])
        Y[1] = _make_pcf(np_dtype, [[0, 2], [2, 10]])
        Z = X * Y
        assert isinstance(Z, tensor_cls)
        assert Z[0].to_numpy()[0, 1] == np_dtype(6)
        assert Z[1].to_numpy()[0, 1] == np_dtype(20)

    def test_imul_tensor_tensor(self, pcf_dtype, np_dtype, tensor_cls):
        X = mpcf.zeros((2,), dtype=pcf_dtype)
        X[0] = _make_pcf(np_dtype, [[0, 2], [2, 4]])
        X[1] = _make_pcf(np_dtype, [[0, 10], [2, 20]])
        Y = mpcf.zeros((2,), dtype=pcf_dtype)
        Y[0] = _make_pcf(np_dtype, [[0, 3], [2, 5]])
        Y[1] = _make_pcf(np_dtype, [[0, 2], [2, 10]])
        X *= Y
        assert X[0].to_numpy()[0, 1] == np_dtype(6)

    def test_truediv_tensor_tensor(self, pcf_dtype, np_dtype, tensor_cls):
        X = mpcf.zeros((2,), dtype=pcf_dtype)
        X[0] = _make_pcf(np_dtype, [[0, 6], [2, 8]])
        X[1] = _make_pcf(np_dtype, [[0, 10], [2, 20]])
        Y = mpcf.zeros((2,), dtype=pcf_dtype)
        Y[0] = _make_pcf(np_dtype, [[0, 2], [2, 2]])
        Y[1] = _make_pcf(np_dtype, [[0, 5], [2, 4]])
        Z = X / Y
        assert isinstance(Z, tensor_cls)
        assert Z[0].to_numpy()[0, 1] == np_dtype(3)
        assert Z[1].to_numpy()[0, 1] == np_dtype(2)

    def test_itruediv_tensor_tensor(self, pcf_dtype, np_dtype, tensor_cls):
        X = mpcf.zeros((2,), dtype=pcf_dtype)
        X[0] = _make_pcf(np_dtype, [[0, 6], [2, 8]])
        X[1] = _make_pcf(np_dtype, [[0, 10], [2, 20]])
        Y = mpcf.zeros((2,), dtype=pcf_dtype)
        Y[0] = _make_pcf(np_dtype, [[0, 2], [2, 2]])
        Y[1] = _make_pcf(np_dtype, [[0, 5], [2, 4]])
        X /= Y
        assert X[0].to_numpy()[0, 1] == np_dtype(3)


# --- Float tensor broadcasting ---


def _check_broadcast_op(np_a, np_b, op, TensorType=mpcf.Float64Tensor):
    """Apply op to both numpy arrays and mpcf tensors, assert results match."""
    X = TensorType(np_a)
    Y = TensorType(np_b)
    expected = op(np_a, np_b)
    result = np.asarray(op(X, Y))
    npt.assert_array_almost_equal(result, expected)


class TestFloat64TensorBroadcast:
    def test_add_same_shape(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 20.0, 30.0])
        _check_broadcast_op(a, b, lambda x, y: x + y)

    def test_add_broadcast_row_vector(self):
        a = np.array([[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]])
        b = np.array([100.0, 200.0, 300.0, 400.0])
        _check_broadcast_op(a, b, lambda x, y: x + y)

    def test_add_broadcast_both_expand(self):
        a = np.array([[1.0], [2.0]])
        b = np.array([[10.0, 20.0, 30.0]])
        _check_broadcast_op(a, b, lambda x, y: x + y)

    def test_sub_broadcast(self):
        a = np.array([[10.0, 20.0], [30.0, 40.0]])
        b = np.array([1.0, 2.0])
        _check_broadcast_op(a, b, lambda x, y: x - y)

    def test_mul_broadcast(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([10.0, 20.0])
        _check_broadcast_op(a, b, lambda x, y: x * y)

    def test_truediv_broadcast(self):
        a = np.array([[10.0, 20.0], [30.0, 40.0]])
        b = np.array([2.0, 5.0])
        _check_broadcast_op(a, b, lambda x, y: x / y)

    def test_iadd_broadcast(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([10.0, 20.0])
        X = mpcf.Float64Tensor(a.copy())
        Y = mpcf.Float64Tensor(b)
        X += Y
        expected = a + b
        npt.assert_array_equal(np.asarray(X), expected)

    def test_iadd_incompatible_raises(self):
        X = mpcf.Float64Tensor(np.array([1.0, 2.0]))
        Y = mpcf.Float64Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        try:
            X += Y
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_incompatible_shapes_raise(self):
        X = mpcf.Float64Tensor(np.array([1.0, 2.0, 3.0]))
        Y = mpcf.Float64Tensor(np.array([1.0, 2.0]))
        try:
            _ = X + Y
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_add_does_not_modify_originals(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 20.0, 30.0])
        X = mpcf.Float64Tensor(a)
        Y = mpcf.Float64Tensor(b)
        _ = X + Y
        npt.assert_array_equal(np.asarray(X), a)
        npt.assert_array_equal(np.asarray(Y), b)


class TestFloat32TensorBroadcast:
    def test_add_broadcast(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([10.0, 20.0], dtype=np.float32)
        _check_broadcast_op(a, b, lambda x, y: x + y, mpcf.Float32Tensor)
