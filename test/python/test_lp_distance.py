"""Tests for lp_distance (scalar distance between two PCFs)."""

import numpy as np
import pytest

import masspcf as mpcf


def _pcf(points, dtype):
    return mpcf.Pcf(np.array(points), dtype=dtype)


# --- Basic correctness ---

def test_lp_distance_identical_functions(pcf_dtype):
    f = _pcf([[0.0, 5.0], [1.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, f) == pytest.approx(0.0)


def test_lp_distance_both_zero(pcf_dtype):
    f = _pcf([[0.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g) == pytest.approx(0.0)


def test_lp_distance_l1_basic(pcf_dtype):
    f = _pcf([[0.0, 5.0], [1.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g) == pytest.approx(5.0)


def test_lp_distance_default_p_is_l1(pcf_dtype):
    f = _pcf([[0.0, 5.0], [1.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g) == pytest.approx(mpcf.lp_distance(f, g, p=1))


def test_lp_distance_returns_float(pcf_dtype):
    f = _pcf([[0.0, 1.0]], pcf_dtype)
    g = _pcf([[0.0, 0.0]], pcf_dtype)
    assert isinstance(mpcf.lp_distance(f, g), float)


# --- Symmetry and metric properties ---

def test_lp_distance_symmetric(pcf_dtype):
    f = _pcf([[0.0, 3.0], [1.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, 1.0], [2.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g) == pytest.approx(mpcf.lp_distance(g, f))


def test_lp_distance_triangle_inequality(pcf_dtype):
    f = _pcf([[0.0, 3.0], [1.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, 1.0], [1.0, 0.0]], pcf_dtype)
    h = _pcf([[0.0, 0.0]], pcf_dtype)
    d_fg = mpcf.lp_distance(f, g)
    d_gh = mpcf.lp_distance(g, h)
    d_fh = mpcf.lp_distance(f, h)
    assert d_fh <= d_fg + d_gh + 1e-10


def test_lp_distance_nonnegative(pcf_dtype):
    f = _pcf([[0.0, 2.0], [1.0, -1.0], [3.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, -3.0], [2.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g) >= 0.0


# --- Lp distances ---

def test_lp_distance_l2(pcf_dtype):
    # f=3 on [0,1), g=0 on [0,1) => L2 = sqrt(9*1) = 3
    f = _pcf([[0.0, 3.0], [1.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, 0.0], [1.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g, p=2) == pytest.approx(3.0)


def test_lp_distance_l3(pcf_dtype):
    # f=4 on [0,1), g=1 on [0,1) => L3 = (|4-1|^3 * 1)^(1/3) = 3
    f = _pcf([[0.0, 4.0], [1.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, 1.0], [1.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g, p=3) == pytest.approx(3.0)


def test_lp_distance_large_p(pcf_dtype):
    # As p -> inf, Lp -> L_inf = max |f-g|. For f=2 on [0,1), g=0: all p give 2.
    f = _pcf([[0.0, 2.0], [1.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, 0.0], [1.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g, p=50) == pytest.approx(2.0, abs=1e-2)


def test_lp_distance_p_equals_1_explicitly(pcf_dtype):
    f = _pcf([[0.0, 2.0], [1.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g, p=1) == pytest.approx(2.0)


# --- Multiple breakpoints ---

def test_lp_distance_multiple_breakpoints(pcf_dtype):
    # f = 3 on [0,1), 1 on [1,2), 0 on [2,inf)
    # g = 1 on [0,1), 1 on [1,2), 0 on [2,inf)
    # L1 = |3-1|*1 + |1-1|*1 = 2
    f = _pcf([[0.0, 3.0], [1.0, 1.0], [2.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g) == pytest.approx(2.0)


def test_lp_distance_different_breakpoints(pcf_dtype):
    # f = 2 on [0,1), 0 on [1,inf)
    # g = 1 on [0,2), 0 on [2,inf)
    # L1 = |2-1|*1 + |0-1|*1 = 2
    f = _pcf([[0.0, 2.0], [1.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, 1.0], [2.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g) == pytest.approx(2.0)


def test_lp_distance_interleaved_breakpoints(pcf_dtype):
    # f = 4 on [0,1), 2 on [1,3), 0 on [3,inf)
    # g = 1 on [0,2), 3 on [2,4), 0 on [4,inf)
    # intervals: [0,1): |4-1|=3, [1,2): |2-1|=1, [2,3): |2-3|=1, [3,4): |0-3|=3
    # L1 = 3*1 + 1*1 + 1*1 + 3*1 = 8
    f = _pcf([[0.0, 4.0], [1.0, 2.0], [3.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, 1.0], [2.0, 3.0], [4.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g) == pytest.approx(8.0)


def test_lp_distance_one_breakpoint_inside_other(pcf_dtype):
    # f = 6 on [0,3), 0 on [3,inf)
    # g = 2 on [0,1), 4 on [1,2), 0 on [2,inf)
    # intervals: [0,1): |6-2|=4, [1,2): |6-4|=2, [2,3): |6-0|=6
    # L1 = 4*1 + 2*1 + 6*1 = 12
    f = _pcf([[0.0, 6.0], [3.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, 2.0], [1.0, 4.0], [2.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g) == pytest.approx(12.0)


def test_lp_distance_no_shared_breakpoints(pcf_dtype):
    # f = 5 on [0, 0.5), 0 on [0.5, inf)
    # g = 3 on [0, 1.5), 0 on [1.5, inf)
    # intervals: [0,0.5): |5-3|=2, [0.5,1.5): |0-3|=3
    # L1 = 2*0.5 + 3*1.0 = 4.0
    f = _pcf([[0.0, 5.0], [0.5, 0.0]], pcf_dtype)
    g = _pcf([[0.0, 3.0], [1.5, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g) == pytest.approx(4.0)


def test_lp_distance_many_breakpoints(pcf_dtype):
    # Staircase: f = i on [i, i+1) for i=0..4
    # g = 0 everywhere
    # L1 = 0*1 + 1*1 + 2*1 + 3*1 + 4*1 = 10
    f = _pcf([[float(i), float(i)] for i in range(5)] + [[5.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g) == pytest.approx(10.0)


# --- Negative values ---

def test_lp_distance_negative_values(pcf_dtype):
    # f = -3 on [0,1), g = 2 on [0,1) => L1 = |-3-2|*1 = 5
    f = _pcf([[0.0, -3.0], [1.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, 2.0], [1.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g) == pytest.approx(5.0)


def test_lp_distance_both_negative(pcf_dtype):
    # f = -1 on [0,1), g = -4 on [0,1) => L1 = |-1-(-4)|*1 = 3
    f = _pcf([[0.0, -1.0], [1.0, 0.0]], pcf_dtype)
    g = _pcf([[0.0, -4.0], [1.0, 0.0]], pcf_dtype)
    assert mpcf.lp_distance(f, g) == pytest.approx(3.0)


# --- Validation ---

def test_lp_distance_rejects_p_less_than_1():
    f = mpcf.Pcf(np.array([[0.0, 1.0]]))
    g = mpcf.Pcf(np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="p must be >= 1"):
        mpcf.lp_distance(f, g, p=0.5)


def test_lp_distance_rejects_p_zero():
    f = mpcf.Pcf(np.array([[0.0, 1.0]]))
    g = mpcf.Pcf(np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="p must be >= 1"):
        mpcf.lp_distance(f, g, p=0)


def test_lp_distance_rejects_p_negative():
    f = mpcf.Pcf(np.array([[0.0, 1.0]]))
    g = mpcf.Pcf(np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="p must be >= 1"):
        mpcf.lp_distance(f, g, p=-2)


def test_lp_distance_rejects_non_pcf():
    f = mpcf.Pcf(np.array([[0.0, 1.0]]))
    with pytest.raises(TypeError):
        mpcf.lp_distance(f, 42)


def test_lp_distance_rejects_mismatched_dtypes():
    f = mpcf.Pcf(np.array([[0.0, 1.0]], dtype=np.float32))
    g = mpcf.Pcf(np.array([[0.0, 1.0]], dtype=np.float64))
    with pytest.raises(TypeError):
        mpcf.lp_distance(f, g)


def test_lp_distance_rejects_integer_pcf():
    f = mpcf.Pcf(np.array([[0, 1], [1, 0]], dtype=np.int32))
    g = mpcf.Pcf(np.array([[0, 0]], dtype=np.int32))
    with pytest.raises(TypeError):
        mpcf.lp_distance(f, g)
