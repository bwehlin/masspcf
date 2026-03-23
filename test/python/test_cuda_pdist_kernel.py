"""Tests for pdist and l2_kernel on the CUDA path.

Skipped when no GPU is available, unless MPCF_REQUIRE_CUDA=1 is set,
in which case missing CUDA causes a hard failure.
"""

import os

import numpy as np
import pytest

import masspcf as mpcf
from masspcf import _mpcf_cpp as cpp
from masspcf.distance_matrix import DistanceMatrix
from masspcf.symmetric_matrix import SymmetricMatrix

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
    """Force the CUDA path by setting threshold to 0, restore after test."""
    cpp.set_cuda_threshold(0)
    yield
    cpp.set_cuda_threshold(500)


@requires_cuda
def test_pdist_cuda_returns_distance_matrix():
    X = mpcf.zeros((3,))
    D = mpcf.pdist(X)
    assert isinstance(D, DistanceMatrix)
    assert D.size == 3


@requires_cuda
def test_pdist_cuda_correct_values():
    X = mpcf.zeros((2,), dtype=mpcf.pcf64)

    X[0] = mpcf.Pcf(np.array([[0.0, 10.0], [2.0, 5.0], [3.0, 0.0]]))
    X[1] = mpcf.Pcf(np.array([[0.0, 5.0], [6.0, 0.0]]))

    D = mpcf.pdist(X)

    assert D.size == 2
    assert D[0, 0] == 0.0
    assert D[0, 1] == pytest.approx(2 * 5 + 3 * 5)
    assert D[1, 0] == D[0, 1]
    assert D[1, 1] == 0.0


@requires_cuda
def test_pdist_cuda_to_dense():
    X = mpcf.zeros((2,), dtype=mpcf.pcf64)

    X[0] = mpcf.Pcf(np.array([[0.0, 10.0], [2.0, 5.0], [3.0, 0.0]]))
    X[1] = mpcf.Pcf(np.array([[0.0, 5.0], [6.0, 0.0]]))

    D = mpcf.pdist(X)
    dense = D.to_dense()

    assert isinstance(dense, np.ndarray)
    assert dense.shape == (2, 2)
    assert dense[0, 1] == pytest.approx(2 * 5 + 3 * 5)


@requires_cuda
def test_l2_kernel_cuda_returns_symmetric_matrix():
    X = mpcf.zeros((3,))
    K = mpcf.l2_kernel(X)
    assert isinstance(K, SymmetricMatrix)
    assert K.size == 3


@requires_cuda
def test_l2_kernel_cuda_correct_values():
    X = mpcf.zeros((2,), dtype=mpcf.pcf64)

    X[0] = mpcf.Pcf(np.array([[0.0, 4.0], [3.0, 0.0]]))
    X[1] = mpcf.Pcf(np.array([[0.0, 2.0], [3.0, 0.0]]))

    K = mpcf.l2_kernel(X)

    assert K.size == 2
    assert K[0, 0] == pytest.approx(48.0)
    assert K[0, 1] == pytest.approx(24.0)
    assert K[1, 0] == K[0, 1]
    assert K[1, 1] == pytest.approx(12.0)


@requires_cuda
def test_l2_kernel_cuda_to_dense():
    X = mpcf.zeros((2,), dtype=mpcf.pcf64)

    X[0] = mpcf.Pcf(np.array([[0.0, 4.0], [3.0, 0.0]]))
    X[1] = mpcf.Pcf(np.array([[0.0, 2.0], [3.0, 0.0]]))

    K = mpcf.l2_kernel(X)
    dense = K.to_dense()

    assert isinstance(dense, np.ndarray)
    assert dense.shape == (2, 2)
    assert dense[0, 0] == pytest.approx(48.0)
    assert dense[0, 1] == pytest.approx(24.0)
