"""Whole-pipeline tests for compute_homological_kernel through the public API.

Expected barcodes are hand-computed. The 2D point-cloud cases use the
orthogonal projection onto the diagonal y = x (the "diagonal" transform
preset): a point (x, y) maps to (m, m) with m = (x + y) / 2.
"""

import numpy as np
import pytest

import stablebear as sb
import stablebear.persistence as pers
from stablebear.persistence.barcode import Barcode

S2 = np.sqrt(2.0)
S10 = np.sqrt(10.0)


def _bc(pairs, dtype=np.float64):
    """Create a Barcode from a list of (birth, death) pairs."""
    if len(pairs) == 0:
        return Barcode(np.zeros((0, 2), dtype=dtype))
    return Barcode(np.array(pairs, dtype=dtype))


def _distance_matrix(points, dtype=sb.float64):
    """Build a DistanceMatrix of Euclidean pairwise distances."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    dm = sb.DistanceMatrix(n, dtype=dtype)
    for i in range(n):
        for j in range(i + 1, n):
            dm[i, j] = float(np.linalg.norm(pts[i] - pts[j]))
    return dm


# --- Hand-computed 2D cases for the diagonal projection ---

# All points on the diagonal are fixed by the projection, so d' = d and
# every bar is empty [w, w).
DIAGONAL_POINTS = [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]]
DIAGONAL_BARS = [[S2, S2], [2 * S2, 2 * S2]]

# (2, -2) projects onto (0, 0), coinciding with the projection of the first
# point: the pair merges at 0 in d', giving a bar born at 0. The third point
# lies on the diagonal, so the second bar is empty.
COINCIDENT_POINTS = [[0.0, 0.0], [2.0, -2.0], [3.0, 3.0]]
COINCIDENT_BARS = [[0.0, 2 * S2], [3 * S2, 3 * S2]]

# As above, but the third point is off the diagonal, so the second bar is
# non-empty as well.
OFF_DIAGONAL_POINTS = [[0.0, 0.0], [2.0, -2.0], [4.0, 2.0]]
OFF_DIAGONAL_BARS = [[0.0, 2 * S2], [3 * S2, np.sqrt(20.0)]]

# Coincidence and merge-order inversion combined: (4, 0) and (0, 4) both
# project onto (2, 2), and the d' merge order differs from the d merge order.
# In d the coincident pair is not directly connected; it joins through
# (3, 3), so its bar dies at the subdominant distance sqrt(10), carried by a
# pair touching neither of the coincident points.
FLAGSHIP_POINTS = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [3.0, 3.0]]
FLAGSHIP_PROJECTED = [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [3.0, 3.0]]
FLAGSHIP_BARS = [[0.0, S10], [S2, S10], [2 * S2, 4.0]]


# --- Hand-computed barcodes through the transform preset ---


@pytest.mark.parametrize(
    "points, expected",
    [
        (DIAGONAL_POINTS, DIAGONAL_BARS),
        (COINCIDENT_POINTS, COINCIDENT_BARS),
        (OFF_DIAGONAL_POINTS, OFF_DIAGONAL_BARS),
        (FLAGSHIP_POINTS, FLAGSHIP_BARS),
    ],
    ids=["diagonal", "coincident", "off-diagonal-third", "coincidence-and-inversion"],
)
def test_diagonal_transform_gives_hand_computed_barcode(points, expected):
    bcs = pers.compute_homological_kernel(np.array(points), transform="diagonal")
    assert bcs.shape == (1,)
    assert bcs[0].is_isomorphic_to(_bc(expected))


# --- Route consistency: explicit X_prime, distance matrices, float32 ---


def test_explicit_x_prime_matches_transform_preset():
    X = np.array(FLAGSHIP_POINTS)
    via_transform = pers.compute_homological_kernel(X, transform="diagonal")
    via_x_prime = pers.compute_homological_kernel(X, np.array(FLAGSHIP_PROJECTED))
    assert via_x_prime[0].is_isomorphic_to(via_transform[0])


def test_distmat_route_matches_pcloud_route():
    via_pcloud = pers.compute_homological_kernel(
        np.array(FLAGSHIP_POINTS), np.array(FLAGSHIP_PROJECTED)
    )
    via_distmat = pers.compute_homological_kernel(
        _distance_matrix(FLAGSHIP_POINTS), _distance_matrix(FLAGSHIP_PROJECTED)
    )
    assert via_distmat.shape == (1,)
    assert via_distmat[0].is_isomorphic_to(via_pcloud[0])


def test_float32_input_gives_float32_barcode():
    X = np.array(FLAGSHIP_POINTS, dtype=np.float32)
    bcs = pers.compute_homological_kernel(X, transform="diagonal")
    assert bcs.dtype == sb.barcode32
    assert bcs[0].is_isomorphic_to(_bc(FLAGSHIP_BARS, dtype=np.float32))


# --- Multi-element tensors ---


def test_multi_element_tensor_computes_per_element():
    X = sb.zeros((2,), dtype=sb.pcloud64)
    X[0] = np.array(FLAGSHIP_POINTS)
    X[1] = np.array(COINCIDENT_POINTS)

    bcs = pers.compute_homological_kernel(X, transform="diagonal")

    assert bcs.shape == (2,)
    assert bcs[0].is_isomorphic_to(_bc(FLAGSHIP_BARS))
    assert bcs[1].is_isomorphic_to(_bc(COINCIDENT_BARS))


# --- Error paths through the public API ---


def test_requires_exactly_one_of_x_prime_and_transform():
    X = np.array(DIAGONAL_POINTS)
    with pytest.raises(ValueError):
        pers.compute_homological_kernel(X)
    with pytest.raises(ValueError):
        pers.compute_homological_kernel(X, X.copy(), transform="diagonal")


def test_unknown_transform_raises():
    with pytest.raises(ValueError):
        pers.compute_homological_kernel(np.array(DIAGONAL_POINTS), transform="nope")


def test_transform_requires_point_cloud_input():
    dm = _distance_matrix(DIAGONAL_POINTS)
    with pytest.raises(TypeError):
        pers.compute_homological_kernel(dm, transform="diagonal")


def test_mixed_input_kinds_raise():
    with pytest.raises(TypeError):
        pers.compute_homological_kernel(
            np.array(DIAGONAL_POINTS), _distance_matrix(DIAGONAL_POINTS)
        )


def test_non_dominated_metric_raises():
    # X' scales distances up, so d' does not stay below d.
    X = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    with pytest.raises(RuntimeError):
        pers.compute_homological_kernel(X, 2.0 * X)


def test_mismatched_point_counts_raise():
    X = np.array(DIAGONAL_POINTS)
    with pytest.raises(RuntimeError):
        pers.compute_homological_kernel(X, X[:2].copy())
