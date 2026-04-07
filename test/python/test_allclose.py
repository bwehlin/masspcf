import numpy as np
import pytest
import masspcf as mpcf


class TestAllcloseFloatTensor:
    def test_identical(self):
        a = mpcf.FloatTensor([1.0, 2.0, 3.0])
        assert mpcf.allclose(a, a)

    def test_equal_copies(self):
        a = mpcf.FloatTensor([1.0, 2.0, 3.0])
        b = a.copy()
        assert mpcf.allclose(a, b)

    def test_within_atol(self):
        a = mpcf.FloatTensor([1.0, 2.0, 3.0])
        b = mpcf.FloatTensor([1.0 + 1e-9, 2.0, 3.0])
        assert mpcf.allclose(a, b)

    def test_outside_atol(self):
        a = mpcf.FloatTensor([1.0, 2.0, 3.0])
        b = mpcf.FloatTensor([1.0, 2.1, 3.0])
        assert not mpcf.allclose(a, b)

    def test_custom_atol(self):
        a = mpcf.FloatTensor([1.0, 2.0, 3.0])
        b = mpcf.FloatTensor([1.0, 2.05, 3.0])
        assert not mpcf.allclose(a, b, atol=0.01)
        assert mpcf.allclose(a, b, atol=0.1)

    def test_multidimensional(self):
        a = mpcf.FloatTensor(np.ones((3, 4), dtype=np.float32))
        b = mpcf.FloatTensor(np.ones((3, 4), dtype=np.float32))
        assert mpcf.allclose(a, b)


class TestAllcloseDistanceMatrix:
    def test_identical(self):
        m = mpcf.DistanceMatrix(5, dtype=mpcf.float64)
        m[0, 1] = 3.14
        assert mpcf.allclose(m, m)

    def test_within_tolerance(self):
        a = mpcf.DistanceMatrix(3, dtype=mpcf.float64)
        b = mpcf.DistanceMatrix(3, dtype=mpcf.float64)
        a[0, 1] = 1.0
        b[0, 1] = 1.0 + 1e-9
        assert mpcf.allclose(a, b)

    def test_outside_tolerance(self):
        a = mpcf.DistanceMatrix(3, dtype=mpcf.float64)
        b = mpcf.DistanceMatrix(3, dtype=mpcf.float64)
        a[0, 1] = 1.0
        b[0, 1] = 2.0
        assert not mpcf.allclose(a, b)


class TestAllcloseSymmetricMatrix:
    def test_identical(self):
        m = mpcf.SymmetricMatrix(5, dtype=mpcf.float64)
        m[0, 1] = 3.14
        assert mpcf.allclose(m, m)

    def test_within_tolerance(self):
        a = mpcf.SymmetricMatrix(3, dtype=mpcf.float64)
        b = mpcf.SymmetricMatrix(3, dtype=mpcf.float64)
        a[0, 1] = 1.0
        b[0, 1] = 1.0 + 1e-9
        assert mpcf.allclose(a, b)


class TestAllcloseErrors:
    def test_mismatched_types(self):
        a = mpcf.FloatTensor([1.0])
        b = mpcf.DistanceMatrix(1, dtype=mpcf.float64)
        with pytest.raises(TypeError, match="same supported type"):
            mpcf.allclose(a, b)

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="same supported type"):
            mpcf.allclose(42, 42)
