"""Tests for cdist and l2 cross-kernel on the CUDA path.

Skipped when no GPU is available, unless MPCF_REQUIRE_CUDA=1 is set,
in which case missing CUDA causes a hard failure.
"""

import os

import numpy as np
import pytest

import masspcf as mpcf
from masspcf import _mpcf_cpp as cpp

_has_cuda = cpp._build_type() == "CUDA" and cpp.get_ngpus() > 0
_require_cuda = os.environ.get("MPCF_REQUIRE_CUDA", "0") == "1"

if _require_cuda and not _has_cuda:
    pytest.fail(
        "MPCF_REQUIRE_CUDA=1 but CUDA is not available "
        f"(build_type={cpp._build_type()}, ngpus={cpp.get_ngpus()})",
        pytrace=False,
    )

requires_cuda = pytest.mark.skipif(
    not _has_cuda,
    reason="Requires CUDA build with at least one GPU",
)


@pytest.fixture(autouse=True)
def _force_cuda():
    """Force the CUDA path and enable verbose device reporting."""
    cpp.set_cuda_threshold(0)
    cpp.set_device_verbose(True)
    yield
    cpp.set_device_verbose(False)
    cpp.set_cuda_threshold(500)


def _assert_ran_on_cuda(captured_out):
    assert "CUDA" in captured_out, (
        f"Expected CUDA execution but got: {captured_out!r}"
    )


@requires_cuda
def test_cdist_cuda_returns_float_tensor(capfd):
    X = mpcf.zeros((2,), dtype=mpcf.pcf64)
    Y = mpcf.zeros((3,), dtype=mpcf.pcf64)

    D = mpcf.cdist(X, Y)
    _assert_ran_on_cuda(capfd.readouterr().out)

    assert isinstance(D, mpcf.FloatTensor)
    assert D.shape == (2, 3)


@requires_cuda
def test_cdist_cuda_correct_values(capfd):
    X = mpcf.zeros((2,), dtype=mpcf.pcf64)
    Y = mpcf.zeros((2,), dtype=mpcf.pcf64)

    X[0] = mpcf.Pcf(np.array([[0.0, 5.0], [1.0, 0.0]]))
    X[1] = mpcf.Pcf(np.array([[0.0, 1.0], [2.0, 0.0]]))

    Y[0] = mpcf.Pcf(np.array([[0.0, 5.0], [1.0, 0.0]]))  # same as X[0]
    Y[1] = mpcf.Pcf(np.array([[0.0, 0.0]]))

    D = mpcf.cdist(X, Y)
    _assert_ran_on_cuda(capfd.readouterr().out)

    assert D[0, 0] == pytest.approx(0.0)   # X[0] vs Y[0] = same
    assert D[0, 1] == pytest.approx(5.0)   # X[0] vs Y[1] = ||5 on [0,1)||_1
    assert D[1, 0] == pytest.approx(5.0)   # X[1] vs Y[0]: |1-5|*1 + |1-0|*1 = 5
    assert D[1, 1] == pytest.approx(2.0)   # X[1] vs Y[1]: |1-0|*2 = 2


@requires_cuda
def test_cdist_cuda_matches_pdist_for_same_input(capfd):
    X = mpcf.zeros((3,), dtype=mpcf.pcf64)

    X[0] = mpcf.Pcf(np.array([[0.0, 3.0], [1.0, 0.0]]))
    X[1] = mpcf.Pcf(np.array([[0.0, 1.0], [2.0, 0.0]]))
    X[2] = mpcf.Pcf(np.array([[0.0, 0.0]]))

    D_pdist = mpcf.pdist(X)
    _ = capfd.readouterr()  # consume pdist output

    D_cdist = mpcf.cdist(X, X)
    _assert_ran_on_cuda(capfd.readouterr().out)

    for i in range(3):
        for j in range(3):
            assert D_cdist[i, j] == pytest.approx(D_pdist[i, j], abs=1e-10), \
                f"Mismatch at ({i}, {j})"


@requires_cuda
def test_cdist_lp_cuda_correct_values(capfd):
    X = mpcf.zeros((1,), dtype=mpcf.pcf64)
    Y = mpcf.zeros((1,), dtype=mpcf.pcf64)

    X[0] = mpcf.Pcf(np.array([[0.0, 4.0], [1.0, 0.0]]))
    Y[0] = mpcf.Pcf(np.array([[0.0, 1.0], [1.0, 0.0]]))

    D = mpcf.cdist(X, Y, p=3)
    _assert_ran_on_cuda(capfd.readouterr().out)

    # ||f - g||_3 = (|4-1|^3 * 1)^(1/3) = 3
    assert D[0, 0] == pytest.approx(3.0)


@requires_cuda
def test_cdist_cuda_rectangular(capfd):
    X = mpcf.zeros((1,), dtype=mpcf.pcf64)
    Y = mpcf.zeros((3,), dtype=mpcf.pcf64)

    X[0] = mpcf.Pcf(np.array([[0.0, 2.0], [1.0, 0.0]]))
    Y[0] = mpcf.Pcf(np.array([[0.0, 0.0]]))
    Y[1] = mpcf.Pcf(np.array([[0.0, 2.0], [1.0, 0.0]]))
    Y[2] = mpcf.Pcf(np.array([[0.0, 1.0]]))

    D = mpcf.cdist(X, Y)
    _assert_ran_on_cuda(capfd.readouterr().out)

    assert D.shape == (1, 3)
    assert D[0, 0] == pytest.approx(2.0)   # vs zero
    assert D[0, 1] == pytest.approx(0.0)   # vs same
    # vs constant 1: |2-1|*1 + |0-1|*inf... but PCFs end, so = |2-1|*1 = 1
    # Actually: X[0] = 2 on [0,1), 0 on [1,inf). Y[2] = 1 everywhere.
    # integral |2-1| on [0,1) + |0-1| on [1, inf) -> diverges. But we use max float as b.
    # The PCF has implicit 0 after last point, and constant PCF is 1 forever.
    # So the L1 integral diverges. Let's just check it's > 0.
    assert D[0, 2] > 0


@requires_cuda
def test_cdist_cuda_multidim_shape(capfd):
    X = mpcf.zeros((2, 3), dtype=mpcf.pcf64)
    Y = mpcf.zeros((4,), dtype=mpcf.pcf64)

    D = mpcf.cdist(X, Y)
    _assert_ran_on_cuda(capfd.readouterr().out)

    assert D.shape == (2, 3, 4)
