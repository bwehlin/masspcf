import numpy as np
import numpy.testing as npt

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


# --- PCF tensors ---


class TestPcf32TensorArithmetic:
    def _make_tensor(self):
        X = mpcf.zeros((2,))
        X[0] = _make_pcf32([[0, 1], [2, 3]])
        X[1] = _make_pcf32([[0, 10], [2, 20]])
        return X

    def test_add_pcf(self):
        X = self._make_tensor()
        pcf = _make_pcf32([[0, 100], [2, 200]])
        Y = X + pcf
        assert isinstance(Y, mpcf.Pcf32Tensor)
        npt.assert_almost_equal(Y[0].to_numpy()[0, 1], 101.0)
        npt.assert_almost_equal(Y[1].to_numpy()[0, 1], 110.0)

    def test_iadd_pcf(self):
        X = self._make_tensor()
        pcf = _make_pcf32([[0, 100], [2, 200]])
        X += pcf
        npt.assert_almost_equal(X[0].to_numpy()[0, 1], 101.0)

    def test_sub_pcf(self):
        X = self._make_tensor()
        pcf = _make_pcf32([[0, 1], [2, 1]])
        Y = X - pcf
        assert isinstance(Y, mpcf.Pcf32Tensor)
        npt.assert_almost_equal(Y[0].to_numpy()[0, 1], 0.0)
        npt.assert_almost_equal(Y[1].to_numpy()[0, 1], 9.0)

    def test_isub_pcf(self):
        X = self._make_tensor()
        pcf = _make_pcf32([[0, 1], [2, 1]])
        X -= pcf
        npt.assert_almost_equal(X[0].to_numpy()[0, 1], 0.0)

    def test_mul_scalar(self):
        X = self._make_tensor()
        Y = X * 2.0
        assert isinstance(Y, mpcf.Pcf32Tensor)
        npt.assert_almost_equal(Y[0].to_numpy()[0, 1], 2.0)
        npt.assert_almost_equal(Y[1].to_numpy()[0, 1], 20.0)

    def test_imul_scalar(self):
        X = self._make_tensor()
        X *= 2.0
        npt.assert_almost_equal(X[0].to_numpy()[0, 1], 2.0)

    def test_truediv_scalar(self):
        X = self._make_tensor()
        Y = X / 2.0
        assert isinstance(Y, mpcf.Pcf32Tensor)
        npt.assert_almost_equal(Y[0].to_numpy()[0, 1], 0.5)
        npt.assert_almost_equal(Y[1].to_numpy()[0, 1], 5.0)

    def test_itruediv_scalar(self):
        X = self._make_tensor()
        X /= 2.0
        npt.assert_almost_equal(X[0].to_numpy()[0, 1], 0.5)

    def test_add_does_not_modify_original(self):
        X = self._make_tensor()
        pcf = _make_pcf32([[0, 100], [2, 200]])
        _ = X + pcf
        npt.assert_almost_equal(X[0].to_numpy()[0, 1], 1.0)

    def test_mul_does_not_modify_original(self):
        X = self._make_tensor()
        _ = X * 10.0
        npt.assert_almost_equal(X[0].to_numpy()[0, 1], 1.0)


class TestPcf64TensorArithmetic:
    def _make_tensor(self):
        X = mpcf.zeros((2,), dtype=mpcf.pcf64)
        X[0] = _make_pcf64([[0, 1], [2, 3]])
        X[1] = _make_pcf64([[0, 10], [2, 20]])
        return X

    def test_add_pcf(self):
        X = self._make_tensor()
        pcf = _make_pcf64([[0, 100], [2, 200]])
        Y = X + pcf
        assert isinstance(Y, mpcf.Pcf64Tensor)
        npt.assert_almost_equal(Y[0].to_numpy()[0, 1], 101.0)

    def test_mul_scalar(self):
        X = self._make_tensor()
        Y = X * 3.0
        assert isinstance(Y, mpcf.Pcf64Tensor)
        npt.assert_almost_equal(Y[0].to_numpy()[0, 1], 3.0)

    def test_truediv_scalar(self):
        X = self._make_tensor()
        Y = X / 4.0
        npt.assert_almost_equal(Y[0].to_numpy()[0, 1], 0.25)


# --- Tensor-Tensor broadcasting ---


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
        except RuntimeError:
            pass

    def test_incompatible_shapes_raise(self):
        X = mpcf.Float64Tensor(np.array([1.0, 2.0, 3.0]))
        Y = mpcf.Float64Tensor(np.array([1.0, 2.0]))
        try:
            _ = X + Y
            assert False, "Should have raised"
        except RuntimeError:
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


class TestPcf32TensorBroadcast:
    def test_add_tensor_tensor_same_shape(self):
        X = mpcf.zeros((2,))
        X[0] = _make_pcf32([[0, 1], [2, 3]])
        X[1] = _make_pcf32([[0, 10], [2, 20]])
        Y = mpcf.zeros((2,))
        Y[0] = _make_pcf32([[0, 100], [2, 200]])
        Y[1] = _make_pcf32([[0, 1000], [2, 2000]])
        Z = X + Y
        assert isinstance(Z, mpcf.Pcf32Tensor)
        npt.assert_almost_equal(Z[0].to_numpy()[0, 1], 101.0)
        npt.assert_almost_equal(Z[1].to_numpy()[0, 1], 1010.0)

    def test_sub_tensor_tensor(self):
        X = mpcf.zeros((2,))
        X[0] = _make_pcf32([[0, 10], [2, 30]])
        X[1] = _make_pcf32([[0, 100], [2, 200]])
        Y = mpcf.zeros((2,))
        Y[0] = _make_pcf32([[0, 1], [2, 1]])
        Y[1] = _make_pcf32([[0, 2], [2, 2]])
        Z = X - Y
        assert isinstance(Z, mpcf.Pcf32Tensor)
        npt.assert_almost_equal(Z[0].to_numpy()[0, 1], 9.0)
        npt.assert_almost_equal(Z[1].to_numpy()[0, 1], 98.0)

    def test_add_broadcast(self):
        X = mpcf.zeros((2, 3))
        for i in range(2):
            for j in range(3):
                X[i, j] = _make_pcf32([[0, float(i * 3 + j)], [2, 0]])
        Y = mpcf.zeros((3,))
        for j in range(3):
            Y[j] = _make_pcf32([[0, 100.0 * (j + 1)], [2, 0]])
        Z = X + Y
        assert Z.shape == (2, 3)
        npt.assert_almost_equal(Z[0, 0].to_numpy()[0, 1], 100.0)
        npt.assert_almost_equal(Z[0, 2].to_numpy()[0, 1], 302.0)
        npt.assert_almost_equal(Z[1, 0].to_numpy()[0, 1], 103.0)


class TestPcf64TensorBroadcast:
    def test_add_tensor_tensor(self):
        X = mpcf.zeros((2,), dtype=mpcf.pcf64)
        X[0] = _make_pcf64([[0, 1], [2, 3]])
        X[1] = _make_pcf64([[0, 10], [2, 20]])
        Y = mpcf.zeros((2,), dtype=mpcf.pcf64)
        Y[0] = _make_pcf64([[0, 100], [2, 200]])
        Y[1] = _make_pcf64([[0, 1000], [2, 2000]])
        Z = X + Y
        assert isinstance(Z, mpcf.Pcf64Tensor)
        npt.assert_almost_equal(Z[0].to_numpy()[0, 1], 101.0)
        npt.assert_almost_equal(Z[1].to_numpy()[0, 1], 1010.0)
