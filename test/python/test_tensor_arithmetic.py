import pytest
import masspcf as mpcf

import numpy as np
import numpy.testing as npt


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
