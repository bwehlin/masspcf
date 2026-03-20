import pytest

import masspcf as mpcf
from masspcf.symmetric_matrix import (
    SymmetricMatrix32Tensor,
    SymmetricMatrix64Tensor,
)
from masspcf.typing import float32, float64, symmat32, symmat64


DTYPES = [symmat32, symmat64]
SCALAR_DTYPES = {symmat32: float32, symmat64: float64}
TENSOR_TYPES = {symmat32: SymmetricMatrix32Tensor, symmat64: SymmetricMatrix64Tensor}


@pytest.fixture(params=DTYPES, ids=["symmat32", "symmat64"])
def dtype(request):
    return request.param


def _make_matrix(n, dtype, values=None):
    scalar_dt = SCALAR_DTYPES[dtype]
    m = mpcf.SymmetricMatrix(n, dtype=scalar_dt)
    if values:
        for (i, j), v in values.items():
            m[i, j] = v
    return m


class TestConstruction:
    def test_zeros(self, dtype):
        t = mpcf.zeros((5,), dtype=dtype)
        assert isinstance(t, TENSOR_TYPES[dtype])
        assert t.shape == mpcf.Shape((5,))

    def test_zeros_multidimensional(self, dtype):
        t = mpcf.zeros((3, 4), dtype=dtype)
        assert t.shape == mpcf.Shape((3, 4))

    def test_element_is_symmetric_matrix(self, dtype):
        t = mpcf.zeros((3,), dtype=dtype)
        elem = t[0]
        assert isinstance(elem, mpcf.SymmetricMatrix)


class TestSetGet:
    def test_set_and_get_element(self, dtype):
        t = mpcf.zeros((3,), dtype=dtype)
        m = _make_matrix(4, dtype, {(0, 1): 5.0, (2, 3): 7.0})
        t[0] = m
        result = t[0]
        assert result[0, 1] == 5.0
        assert result[2, 3] == 7.0

    def test_set_and_get_multidimensional(self, dtype):
        t = mpcf.zeros((2, 3), dtype=dtype)
        m = _make_matrix(3, dtype, {(1, 2): 42.0})
        t[1, 2] = m
        result = t[1, 2]
        assert result[1, 2] == 42.0

    def test_elements_are_independent(self, dtype):
        t = mpcf.zeros((2,), dtype=dtype)
        m0 = _make_matrix(3, dtype, {(0, 1): 1.0})
        m1 = _make_matrix(3, dtype, {(0, 1): 2.0})
        t[0] = m0
        t[1] = m1
        assert t[0][0, 1] == 1.0
        assert t[1][0, 1] == 2.0


class TestSlicing:
    def test_slice_1d(self, dtype):
        t = mpcf.zeros((5,), dtype=dtype)
        for i in range(5):
            m = _make_matrix(2, dtype, {(0, 0): float(i)})
            t[i] = m
        s = t[1:4]
        assert s.shape == mpcf.Shape((3,))
        assert s[0][0, 0] == 1.0
        assert s[2][0, 0] == 3.0

    def test_slice_2d(self, dtype):
        t = mpcf.zeros((3, 4), dtype=dtype)
        m = _make_matrix(2, dtype, {(0, 1): 99.0})
        t[1, 2] = m
        s = t[1:2, 1:3]
        assert s.shape == mpcf.Shape((1, 2))
        assert s[0, 1][0, 1] == 99.0


class TestCopy:
    def test_copy_preserves_values(self, dtype):
        t = mpcf.zeros((3,), dtype=dtype)
        m = _make_matrix(4, dtype, {(1, 2): 3.5})
        t[1] = m
        t2 = t.copy()
        assert t2[1][1, 2] == 3.5

    def test_copy_is_independent(self, dtype):
        t = mpcf.zeros((2,), dtype=dtype)
        m = _make_matrix(3, dtype, {(0, 0): 10.0})
        t[0] = m
        t2 = t.copy()
        m2 = _make_matrix(3, dtype, {(0, 0): 99.0})
        t2[0] = m2
        assert t[0][0, 0] == 10.0
        assert t2[0][0, 0] == 99.0


class TestEquality:
    def test_equal_tensors(self, dtype):
        t1 = mpcf.zeros((3,), dtype=dtype)
        t2 = mpcf.zeros((3,), dtype=dtype)
        m = _make_matrix(2, dtype, {(0, 1): 1.0})
        t1[0] = m
        t2[0] = m
        assert t1.array_equal(t2)

    def test_unequal_tensors(self, dtype):
        t1 = mpcf.zeros((3,), dtype=dtype)
        t2 = mpcf.zeros((3,), dtype=dtype)
        m1 = _make_matrix(2, dtype, {(0, 1): 1.0})
        m2 = _make_matrix(2, dtype, {(0, 1): 2.0})
        t1[0] = m1
        t2[0] = m2
        assert not t1.array_equal(t2)


class TestFlatten:
    def test_flatten(self, dtype):
        t = mpcf.zeros((2, 3), dtype=dtype)
        m = _make_matrix(2, dtype, {(0, 1): 7.0})
        t[1, 2] = m
        flat = t.flatten()
        assert flat.shape == mpcf.Shape((6,))
        assert flat[5][0, 1] == 7.0
