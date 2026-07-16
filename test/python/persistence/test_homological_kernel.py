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


def test_coordinate_transform_gives_hand_computed_barcode():
    # "coordinate" drops the last coordinate, so for 2D input d' = |x_i - x_j|.
    # Points A=(0,0), B=(1,2), C=(3,1): both d MST edges (A-B, B-C) weigh
    # sqrt(5), so every cophenetic distance is sqrt(5). d' merges (A,B)@1,
    # then ({A,B},C)@2.
    points = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
    s5 = np.sqrt(5.0)

    bcs = pers.compute_homological_kernel(points, transform="coordinate")
    assert bcs[0].is_isomorphic_to(_bc([[1.0, s5], [2.0, s5]]))

    # The preset must agree with handing in the projected cloud explicitly.
    projected = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    via_x_prime = pers.compute_homological_kernel(points, projected)
    assert via_x_prime[0].is_isomorphic_to(bcs[0])


# --- Deaths are computed against the quotient space ---
#
# Contracting a d-far pair (which every nonempty bar does) creates a shortcut
# through the contracted component that can lower the death of a LATER merge
# between two other components.


def test_contraction_shortcut_lowers_later_death():
    # Diagonal frame: 1=(-1,11) and 3=(5,5) project to the same point and
    # contract at 0. The merge (0,2)@2*S2 then dies at 5*S2 via the chain
    # 0 -5*S2- 1 ≡ 3 -3*S2- 2, through a component containing neither 0 nor 2.
    points = np.array([[-6.0, 6.0], [-1.0, 11.0], [2.0, 2.0], [5.0, 5.0]])
    bcs = pers.compute_homological_kernel(points, transform="diagonal")
    assert bcs[0].is_isomorphic_to(
        _bc([[0.0, 6 * S2], [2 * S2, 5 * S2], [3 * S2, 3 * S2]])
    )


def test_staircase_death_needs_chained_contraction_shortcuts():
    # Two stacked rungs contract at 0; the (A,B)@1 merge dies at 8 via the
    # two-hop chain A -8- R1 ≡ R1' -3- R2 ≡ R2' -4- B (a k-rung staircase
    # needs k hops). The deaths equal the d-MST weight multiset
    # {3, 4, 8, 10, 10}.
    points = np.array(
        [[0.0, 20.0], [8.0, 20.0], [8.0, 10.0], [5.0, 10.0], [5.0, 0.0], [1.0, 0.0]]
    )
    bcs = pers.compute_homological_kernel(points, transform="coordinate")
    assert bcs[0].is_isomorphic_to(
        _bc([[0.0, 10.0], [0.0, 10.0], [1.0, 8.0], [3.0, 3.0], [4.0, 4.0]])
    )


# --- Randomized cross-check against an independent reference implementation ---


def _subdominant(D):
    """Subdominant (minimax) ultrametric of a dissimilarity matrix."""
    U = D.copy()
    for k in range(len(D)):
        U = np.minimum(U, np.maximum(U[:, k : k + 1], U[k : k + 1, :]))
    np.fill_diagonal(U, 0.0)
    return U


def _single_linkage_merges(D):
    """The n-1 single-linkage merges of D as (i, j, scale), ascending."""
    n = len(D)
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    min_d = D[0].copy()
    parent = np.zeros(n, dtype=int)
    edges = []
    for _ in range(n - 1):
        j = int(np.argmin(np.where(in_tree, np.inf, min_d)))
        edges.append((int(parent[j]), j, float(min_d[j])))
        in_tree[j] = True
        closer = D[j] < min_d
        min_d = np.where(closer, D[j], min_d)
        parent = np.where(closer, j, parent)
    edges.sort(key=lambda e: e[2])
    return edges


def _reference_kernel_barcode(D, Dp):
    """O(n^3) dense reference for ker mu: replays the d' merges while
    maintaining the full subdominant ultrametric of the quotient space
    explicitly (Corollary 4.1.1). Independent of the C++ sweep."""
    n = len(D)
    c = _subdominant(D)
    uf = list(range(n))

    def find(i):
        while uf[i] != i:
            uf[i] = uf[uf[i]]
            i = uf[i]
        return i

    alive = np.ones(n, dtype=bool)
    bars = []
    for a, b, w in _single_linkage_merges(Dp):
        ra, rb = find(a), find(b)
        bars.append([w, max(float(c[ra, rb]), w)])
        row = np.minimum(c[ra], c[rb])
        alive[rb] = False
        uf[rb] = ra
        c[ra], c[:, ra] = row, row
        c[ra, ra] = 0.0
        others = np.where(alive & (np.arange(n) != ra))[0]
        if len(others) > 1:
            sub = c[np.ix_(others, others)]
            c[np.ix_(others, others)] = np.minimum(
                sub, np.maximum(row[others][:, None], row[others][None, :])
            )
    return bars


def test_random_clouds_match_reference_implementation():
    rng = np.random.default_rng(20260716)
    for _ in range(25):
        n = int(rng.integers(3, 13))
        points = rng.normal(size=(n, 2))
        m = points.mean(axis=1)
        D = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        Dp = np.sqrt(2.0) * np.abs(m[:, None] - m[None, :])
        expected = _reference_kernel_barcode(D, np.minimum(Dp, D))

        bcs = pers.compute_homological_kernel(points, transform="diagonal")
        assert bcs[0].is_isomorphic_to(_bc(expected))


def test_random_distance_matrices_match_reference_implementation():
    rng = np.random.default_rng(20260716)
    for _ in range(25):
        n = int(rng.integers(3, 13))
        D = rng.uniform(0.1, 1.1, size=(n, n))
        D = (D + D.T) / 2.0
        np.fill_diagonal(D, 0.0)
        scale = rng.uniform(0.0, 1.0, size=(n, n))
        Dp = D * ((scale + scale.T) / 2.0)
        expected = _reference_kernel_barcode(D, Dp)

        d = sb.DistanceMatrix(n)
        d_prime = sb.DistanceMatrix(n)
        for i in range(n):
            for j in range(i + 1, n):
                d[i, j] = D[i, j]
                d_prime[i, j] = Dp[i, j]
        bcs = pers.compute_homological_kernel(d, d_prime)
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


# --- Small-n edge cases ---


def test_two_points_give_single_bar():
    # The minimal nontrivial input: exactly one merge, one bar.
    points = np.array([[0.0, 0.0], [3.0, 4.0]])  # d = 5, projected gap m = 3.5
    bcs = pers.compute_homological_kernel(points, transform="diagonal")
    assert bcs[0].is_isomorphic_to(_bc([[S2 * 3.5, 5.0]]))

    d = sb.DistanceMatrix(2)
    d[0, 1] = 5.0
    d_prime = sb.DistanceMatrix(2)
    d_prime[0, 1] = 2.5
    bcs = pers.compute_homological_kernel(d, d_prime)
    assert bcs[0].is_isomorphic_to(_bc([[2.5, 5.0]]))


def test_single_point_gives_empty_barcode():
    bcs = pers.compute_homological_kernel(np.array([[1.0, 2.0]]), transform="diagonal")
    assert bcs.shape == (1,)
    assert len(bcs[0]) == 0


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


def test_dtype_mismatch_raises():
    X = np.array(DIAGONAL_POINTS, dtype=np.float32)
    X_prime = np.array(DIAGONAL_POINTS, dtype=np.float64)
    with pytest.raises(TypeError, match="same dtype"):
        pers.compute_homological_kernel(X, X_prime)


def test_outer_shape_mismatch_raises():
    X = sb.zeros((2,), dtype=sb.pcloud64)
    X_prime = sb.zeros((3,), dtype=sb.pcloud64)
    with pytest.raises(ValueError, match="same shape"):
        pers.compute_homological_kernel(X, X_prime)


def test_non_dominated_metric_raises():
    # X' scales distances up, so d' does not stay below d.
    X = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    with pytest.raises(RuntimeError):
        pers.compute_homological_kernel(X, 2.0 * X)


def test_mismatched_point_counts_raise():
    X = np.array(DIAGONAL_POINTS)
    with pytest.raises(RuntimeError):
        pers.compute_homological_kernel(X, X[:2].copy())


def test_rank1_arrays_raise():
    # A 1-D array is not an (n, dim) point cloud and must be rejected.
    X = np.array([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="unexpected shape"):
        pers.compute_homological_kernel(X, X / 2.0)
