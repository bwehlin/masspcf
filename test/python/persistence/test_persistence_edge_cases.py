"""Edge case tests for persistent homology computation."""

import numpy as np
import pytest

import masspcf as mpcf
import masspcf.persistence as mpers


# --- Single point ---
# NOTE: Ripser segfaults on single-point inputs. These tests document the
# expected behavior and are marked xfail until the upstream issue is fixed.


@pytest.mark.xfail(reason="Ripser segfaults on single-point input", run=False)
def test_single_point_has_one_h0_bar():
    """A single point should produce exactly one H0 bar [0, inf) (unreduced)."""
    X = np.array([[1.0, 2.0]])  # 1 point in R^2
    bcs = mpers.compute_persistent_homology(X, maxDim=1, verbose=False)
    h0 = bcs[0]
    h1 = bcs[1]

    assert len(h0) == 1
    bars = h0.to_numpy()
    assert bars[0, 0] == pytest.approx(0.0)
    assert np.isinf(bars[0, 1])

    assert len(h1) == 0


@pytest.mark.xfail(reason="Ripser segfaults on single-point input", run=False)
def test_single_point_reduced_has_no_bars():
    """Reduced homology of a single point should be trivial."""
    X = np.array([[0.0, 0.0]])
    bcs = mpers.compute_persistent_homology(X, maxDim=1, reduced=True, verbose=False)
    h0 = bcs[0]
    assert len(h0) == 0


# --- Two points ---


def test_two_points_h0():
    """Two points produce 2 H0 bars (unreduced): one essential, one finite."""
    X = np.array([[0.0, 0.0], [3.0, 0.0]])  # distance = 3
    bcs = mpers.compute_persistent_homology(X, maxDim=1, verbose=False)
    h0 = bcs[0]
    h1 = bcs[1]

    assert len(h0) == 2
    bars = h0.to_numpy()
    births = sorted(bars[:, 0])
    assert all(b == pytest.approx(0.0) for b in births)

    # One bar should die at distance 3, one should be infinite
    deaths = sorted(bars[:, 1])
    assert deaths[0] == pytest.approx(3.0)
    assert np.isinf(deaths[1])

    assert len(h1) == 0


# --- All identical points ---


@pytest.mark.xfail(reason="Ripser may segfault on degenerate (all-zero distance) inputs", run=False)
def test_identical_points():
    """All identical points: distances are 0, so all merge at scale 0."""
    X = np.array([[1.0, 1.0]] * 5)
    bcs = mpers.compute_persistent_homology(X, maxDim=1, verbose=False)
    h0 = bcs[0]

    bars = h0.to_numpy()
    # All finite bars should die at 0
    finite_bars = bars[~np.isinf(bars[:, 1])]
    for bar in finite_bars:
        assert bar[1] == pytest.approx(0.0)


# --- maxDim variations ---


def test_maxdim_0_only_returns_h0():
    """maxDim=0 should return only H0."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    bcs = mpers.compute_persistent_homology(X, maxDim=0, verbose=False)
    assert bcs.shape == (1,)


def test_maxdim_2_returns_three_dims():
    """maxDim=2 returns H0, H1, H2."""
    X = np.random.randn(10, 3).astype(np.float64)
    bcs = mpers.compute_persistent_homology(X, maxDim=2, verbose=False)
    assert bcs.shape == (3,)


# --- Collinear points ---


def test_collinear_points_no_h1():
    """Collinear points should produce no H1 features."""
    X = np.array([[float(i), 0.0] for i in range(5)])
    bcs = mpers.compute_persistent_homology(X, maxDim=1, verbose=False)
    h1 = bcs[1]
    assert len(h1) == 0


# --- Float32 vs Float64 ---


def test_persistence_f32_and_f64_give_isomorphic_barcodes():
    """Both precisions should give the same topological result."""
    np.random.seed(123)
    pts = np.random.randn(8, 2)

    bcs32 = mpers.compute_persistent_homology(
        pts.astype(np.float32), maxDim=1, verbose=False
    )
    bcs64 = mpers.compute_persistent_homology(
        pts.astype(np.float64), maxDim=1, verbose=False
    )

    # Same number of bars in each dimension
    assert len(bcs32[0]) == len(bcs64[0])
    assert len(bcs32[1]) == len(bcs64[1])


# --- PointCloudTensor with single point per cloud ---


@pytest.mark.xfail(reason="Ripser segfaults on single-point input", run=False)
def test_pcloud_tensor_single_point_per_cloud():
    """Each point cloud has 1 point: trivial topology."""
    X = mpcf.zeros((3,), dtype=mpcf.pcloud64)
    for i in range(3):
        X[i] = np.array([[float(i), 0.0]])

    bcs = mpers.compute_persistent_homology(X, maxDim=1, verbose=False)
    assert bcs.shape == (3, 2)

    for i in range(3):
        assert len(bcs[i, 0]) == 1  # one essential H0 bar
        assert len(bcs[i, 1]) == 0  # no H1
