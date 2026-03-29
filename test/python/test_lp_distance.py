"""Tests for lp_distance (scalar distance between two PCFs)."""

import numpy as np
import pytest

import masspcf as mpcf


def test_lp_distance_identical_functions():
    f = mpcf.Pcf(np.array([[0.0, 5.0], [1.0, 0.0]]))
    assert mpcf.lp_distance(f, f) == pytest.approx(0.0)


def test_lp_distance_l1_basic():
    f = mpcf.Pcf(np.array([[0.0, 5.0], [1.0, 0.0]]))
    g = mpcf.Pcf(np.array([[0.0, 0.0]]))
    assert mpcf.lp_distance(f, g) == pytest.approx(5.0)


def test_lp_distance_default_p_is_l1():
    f = mpcf.Pcf(np.array([[0.0, 5.0], [1.0, 0.0]]))
    g = mpcf.Pcf(np.array([[0.0, 0.0]]))
    assert mpcf.lp_distance(f, g) == pytest.approx(mpcf.lp_distance(f, g, p=1))


def test_lp_distance_lp():
    # f=4 on [0,1), g=1 on [0,1) => L3 distance = (|4-1|^3 * 1)^(1/3) = 3
    f = mpcf.Pcf(np.array([[0.0, 4.0], [1.0, 0.0]]))
    g = mpcf.Pcf(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert mpcf.lp_distance(f, g, p=3) == pytest.approx(3.0)


def test_lp_distance_symmetric():
    f = mpcf.Pcf(np.array([[0.0, 3.0], [1.0, 0.0]]))
    g = mpcf.Pcf(np.array([[0.0, 1.0], [2.0, 0.0]]))
    assert mpcf.lp_distance(f, g) == pytest.approx(mpcf.lp_distance(g, f))


def test_lp_distance_returns_float():
    f = mpcf.Pcf(np.array([[0.0, 1.0]]))
    g = mpcf.Pcf(np.array([[0.0, 0.0]]))
    result = mpcf.lp_distance(f, g)
    assert isinstance(result, float)


def test_lp_distance_rejects_p_less_than_1():
    f = mpcf.Pcf(np.array([[0.0, 1.0]]))
    g = mpcf.Pcf(np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="p must be >= 1"):
        mpcf.lp_distance(f, g, p=0.5)


def test_lp_distance_float32():
    f = mpcf.Pcf(np.array([[0.0, 5.0], [1.0, 0.0]], dtype=np.float32))
    g = mpcf.Pcf(np.array([[0.0, 0.0]], dtype=np.float32))
    assert mpcf.lp_distance(f, g) == pytest.approx(5.0, abs=1e-5)
