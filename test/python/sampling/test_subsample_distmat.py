import numpy as np
import pytest

import stablebear as sb
from stablebear.persistence import barcode_to_stable_rank, compute_persistent_homology
from stablebear.reductions import mean
from stablebear.sampling import Gaussian, Uniform, subsample_relative


def _ref_distmat(n, dim=4, seed=0, np_float=np.float64):
    """A valid (symmetric, zero-diagonal, nonnegative) distance matrix."""
    pts = np.random.default_rng(seed).standard_normal((n, dim)).astype(np_float)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).astype(np_float)
    np.fill_diagonal(d, 0.0)
    return d


@pytest.fixture(params=[(np.float32, sb.distmat32), (np.float64, sb.distmat64)],
                ids=["f32", "f64"])
def float_kind(request):
    return request.param


def test_distmat_output_shape_and_dtype(float_kind):
    np_float, distmat_dtype = float_kind
    D = _ref_distmat(40, np_float=np_float)
    dm = sb.DistanceMatrix.from_dense(D)

    query = np.array([0, 5, 10], dtype=np.uint64)
    subs = subsample_relative(dm, query, sample_size=12, n_instances=7,
                              generator=sb.random.Generator(0))

    assert isinstance(subs, sb.DistanceMatrixTensor)
    assert subs.dtype == distmat_dtype
    assert subs.shape == (3, 7)
    assert subs[0, 0].size == 12
    assert subs[2, 6].size == 12


def test_distmat_query_none_uses_all_rows():
    D = _ref_distmat(20)
    dm = sb.DistanceMatrix.from_dense(D)
    subs = subsample_relative(dm, sample_size=5, n_instances=3,
                              generator=sb.random.Generator(0))
    assert subs.shape == (20, 3)   # one row per reference point


def test_distmat_subsamples_are_indexed_views():
    D = _ref_distmat(50)
    dm = sb.DistanceMatrix.from_dense(D)
    subs = subsample_relative(dm, sample_size=10, n_instances=4,
                              distribution=Gaussian(mean=0.0, sigma=1.0),
                              generator=sb.random.Generator(0))
    el = subs[0, 0]
    assert el.is_indexed
    idx = np.asarray(el.indices)
    assert idx.shape == (10,)
    # The materialized principal submatrix is exactly D over the drawn indices
    # (no entries copied until materialization).
    assert np.allclose(el.materialize().to_dense(), D[np.ix_(idx, idx)])


def test_distmat_zero_weight_points_never_sampled():
    # A hard distance cutoff (Uniform disk): reference points farther than the
    # cutoff from the query row must never appear in any subsample.
    D = _ref_distmat(40, seed=1)
    dm = sb.DistanceMatrix.from_dense(D)
    q = 7
    cutoff = np.median(D[q])

    subs = subsample_relative(dm, np.array([q], dtype=np.uint64),
                              sample_size=5, n_instances=80,
                              distribution=Uniform(high=cutoff),
                              generator=sb.random.Generator(0))

    seen = set()
    for j in range(subs.shape[1]):
        seen.update(int(i) for i in np.asarray(subs[0, j].indices))
    assert seen
    assert all(D[q, i] <= cutoff for i in seen)


def test_distmat_negative_query_indices_match_positive():
    # Negative indices count from the end, as in NumPy — shared with the
    # point-cloud path but asserted here too since the distmat path resolves
    # them into a different backend call.
    D = _ref_distmat(25)
    dm = sb.DistanceMatrix.from_dense(D)

    neg = subsample_relative(dm, np.array([-1, -25]), sample_size=5, n_instances=3,
                             generator=sb.random.Generator(4))
    pos = subsample_relative(dm, np.array([24, 0]), sample_size=5, n_instances=3,
                             generator=sb.random.Generator(4))
    for i in range(2):
        for j in range(3):
            assert np.array_equal(np.asarray(neg[i, j].indices),
                                  np.asarray(pos[i, j].indices))


def test_distmat_gaussian_far_shell_samples_nearest_not_empty():
    # Every distance in the query row lies hundreds of sigmas from the
    # Gaussian mean: a raw exp of the exponents would underflow the whole row
    # to zero (an artificial empty region), while the log-space evaluation
    # must still sample the point whose distance is nearest the mean. This is
    # the distmat twin of test_gaussian_far_query_samples_nearest_point_not_empty.
    pts = np.array([[0.0], [100.0], [101.0], [102.0]])
    D = np.abs(pts - pts.T)
    dm = sb.DistanceMatrix.from_dense(D)

    subs = subsample_relative(dm, np.array([0]), sample_size=5, n_instances=4,
                              distribution=Gaussian(mean=90.0, sigma=0.1),
                              generator=sb.random.Generator(0))

    # Seed-independent: after the max shift the other points sit at -1050 or
    # below, which exp underflows to a hard zero, leaving the row (0, 1, 0, 0).
    # Only holds while sigma < ~0.12 (f64) / ~0.32 (f32).
    for j in range(subs.shape[1]):
        el = subs[0, j]
        assert el.size == 5  # non-empty: repeats of the nearest-to-shell point
        assert set(int(i) for i in np.asarray(el.indices)) == {1}


def test_distmat_per_query_points_differ():
    # Two different query rows should pull from different neighbourhoods.
    D = _ref_distmat(60, seed=2)
    dm = sb.DistanceMatrix.from_dense(D)
    qa, qb = 3, 40

    subs = subsample_relative(dm, np.array([qa, qb], dtype=np.uint64),
                              sample_size=10, n_instances=60,
                              distribution=Gaussian(mean=0.0, sigma=0.5),
                              generator=sb.random.Generator(0))

    drawn = []
    for row in range(2):
        idxs = []
        for j in range(subs.shape[1]):
            idxs.extend(int(i) for i in np.asarray(subs[row, j].indices))
        drawn.append(idxs)

    # The two query points' subsamples should not be identical sets.
    assert set(drawn[0]) != set(drawn[1])


def test_distmat_consecutive_calls_advance_the_generator():
    # The distmat path shares the draw pipeline with the point-cloud path:
    # consecutive calls with one generator must draw fresh samples, and
    # reseeding must replay the sequence call by call.
    D = _ref_distmat(40)
    dm = sb.DistanceMatrix.from_dense(D)
    query = np.array([0, 5, 10], dtype=np.uint64)

    def draw(gen):
        subs = subsample_relative(dm, query, sample_size=10, n_instances=5,
                                  generator=gen)
        return [np.asarray(subs[i, j].indices)
                for i in range(subs.shape[0]) for j in range(subs.shape[1])]

    gen = sb.random.Generator(7)
    a, b = draw(gen), draw(gen)
    assert not all(np.array_equal(x, y) for x, y in zip(a, b))

    gen2 = sb.random.Generator(7)
    a2, b2 = draw(gen2), draw(gen2)
    assert all(np.array_equal(x, y) for x, y in zip(a, a2))
    assert all(np.array_equal(x, y) for x, y in zip(b, b2))


def test_distmat_without_replacement_distinct():
    D = _ref_distmat(15)
    dm = sb.DistanceMatrix.from_dense(D)
    subs = subsample_relative(dm, np.array([0], dtype=np.uint64),
                              sample_size=10, n_instances=20,
                              distribution=Uniform(), replace=False,
                              generator=sb.random.Generator(0))
    for j in range(subs.shape[1]):
        idx = np.asarray(subs[0, j].indices)
        assert len(set(int(i) for i in idx)) == len(idx)


def test_distmat_without_replacement_larger_than_reference_draws_all_points():
    # sample_size is a maximum: without replacement, asking for more points
    # than the reference holds yields every reference point once.
    D = _ref_distmat(5)
    dm = sb.DistanceMatrix.from_dense(D)

    subs = subsample_relative(dm, np.array([0], dtype=np.uint64),
                              sample_size=6, n_instances=4,
                              distribution=Uniform(), replace=False,
                              generator=sb.random.Generator(0))

    for j in range(subs.shape[1]):
        idx = np.asarray(subs[0, j].indices)
        assert sorted(idx) == list(range(5))


def test_distmat_without_replacement_small_region_gives_ragged_subsamples():
    # A Uniform disk holding fewer points than sample_size: each subsample
    # shrinks to exactly the eligible points (ragged, 0 < size < sample_size).
    D = _ref_distmat(40, seed=1)
    dm = sb.DistanceMatrix.from_dense(D)
    q = 7
    cutoff = np.sort(D[q])[3]  # the query row itself + its 3 nearest points
    eligible = {i for i in range(40) if D[q, i] <= cutoff}

    subs = subsample_relative(dm, np.array([q], dtype=np.uint64),
                              sample_size=10, n_instances=15,
                              distribution=Uniform(high=cutoff), replace=False,
                              generator=sb.random.Generator(0))

    for j in range(subs.shape[1]):
        idx = np.asarray(subs[0, j].indices)
        assert set(int(i) for i in idx) == eligible
        assert subs[0, j].size == len(eligible)


@pytest.mark.parametrize("replace", [True, False], ids=["replace", "no-replace"])
def test_distmat_empty_region_gives_empty_subsamples_and_warns(replace):
    # A band beyond the largest distance in the matrix contains no points (the
    # zero self-distance falls below it too): length-0 subsamples, and with
    # verbose=True a warning names the affected query.
    D = _ref_distmat(20)
    dm = sb.DistanceMatrix.from_dense(D)
    lo = D.max() + 1.0

    with pytest.warns(UserWarning, match="1 query point"):
        subs = subsample_relative(dm, np.array([0], dtype=np.uint64),
                                  sample_size=5, n_instances=3,
                                  distribution=Uniform(low=lo, high=lo + 1.0),
                                  replace=replace,
                                  generator=sb.random.Generator(0), verbose=True)

    for j in range(subs.shape[1]):
        el = subs[0, j]
        assert el.is_indexed
        assert el.size == 0
        assert np.asarray(el.indices).size == 0


def test_distmat_tensor_of_one_is_accepted():
    D = _ref_distmat(25)
    dm = sb.DistanceMatrixTensor.from_numpy([D])    # shape (1,)
    subs = subsample_relative(dm, sample_size=6, n_instances=3,
                              generator=sb.random.Generator(0))
    assert isinstance(subs, sb.DistanceMatrixTensor)
    assert subs.shape == (25, 3)


def test_distmat_bad_query_raises():
    D = _ref_distmat(10)
    dm = sb.DistanceMatrix.from_dense(D)
    with pytest.raises(ValueError):
        subsample_relative(dm, np.array([10], dtype=np.uint64),  # out of range
                           sample_size=3, n_instances=1)
    with pytest.raises(ValueError):
        subsample_relative(dm, np.array([], dtype=np.uint64),    # empty
                           sample_size=3, n_instances=1)


def test_distmat_query_by_int_list():
    # A plain Python integer list of row indices is accepted as the query.
    D = _ref_distmat(30)
    dm = sb.DistanceMatrix.from_dense(D)
    subs = subsample_relative(dm, [0, 5, 10], sample_size=6, n_instances=3,
                              generator=sb.random.Generator(0))
    assert subs.shape == (3, 3)


def test_distmat_coordinate_query_raises():
    # A 2-D coordinate query is meaningless for a distance matrix.
    D = _ref_distmat(20)
    dm = sb.DistanceMatrix.from_dense(D)
    coords = np.random.default_rng(0).standard_normal((3, 4))
    with pytest.raises(ValueError):
        subsample_relative(dm, coords, sample_size=3, n_instances=1)


def test_distmat_multi_matrix_reference_raises():
    D = _ref_distmat(8)
    dm = sb.DistanceMatrixTensor.from_numpy([D, D])   # two matrices
    with pytest.raises(ValueError):
        subsample_relative(dm, sample_size=3, n_instances=1)


def test_distmat_copy_keep_source_controls_whether_the_source_is_shared():
    # copy() copies a view's index array but keeps sharing the source buffer;
    # copy(keep_source=False) deep-copies the source too, so a later write to
    # the reference matrix cannot reach it.
    D = _ref_distmat(20, seed=3)
    dm = sb.DistanceMatrix.from_dense(D)
    subs = subsample_relative(dm, np.array([0], dtype=np.uint64),
                              sample_size=6, n_instances=1,
                              generator=sb.random.Generator(0))

    view = subs[0, 0]
    shared = view.copy()
    detached = view.copy(keep_source=False)
    assert shared.is_indexed
    assert detached.is_indexed
    assert np.allclose(shared.to_dense(), view.to_dense())
    assert np.allclose(detached.to_dense(), view.to_dense())

    # Move a source entry the subsample actually selects (two distinct drawn
    # points — sampling is with replacement, so positions may repeat).
    idx = np.asarray(view.indices)
    p, q = next((p, q) for p in range(len(idx)) for q in range(p + 1, len(idx))
                if idx[p] != idx[q])
    a, b = int(idx[p]), int(idx[q])
    dm[a, b] = 42.0
    assert shared[p, q] == 42.0
    assert detached[p, q] == pytest.approx(D[a, b])


def test_distmat_indexed_view_write_detaches():
    # Writing to an indexed view is copy-on-write: the view detaches
    # (materializes in place), so the write succeeds, the matrix is no longer
    # indexed, and the shared reference matrix is untouched.
    D = _ref_distmat(20, seed=3)
    dm = sb.DistanceMatrix.from_dense(D)
    subs = subsample_relative(dm, np.array([0], dtype=np.uint64),
                              sample_size=6, n_instances=1,
                              generator=sb.random.Generator(0))

    view = subs[0, 0]
    assert view.is_indexed
    expected = view.to_dense()

    view[0, 1] = 42.0
    assert not view.is_indexed
    assert view[0, 1] == 42.0
    expected[0, 1] = expected[1, 0] = 42.0
    assert np.allclose(view.to_dense(), expected)
    assert np.allclose(dm.to_dense(), D)


def test_distmat_pipeline_to_relative_stable_rank():
    D = _ref_distmat(120, seed=5)
    dm = sb.DistanceMatrix.from_dense(D)
    query = np.array([0, 30, 60, 90], dtype=np.uint64)

    subs = subsample_relative(dm, query, sample_size=25, n_instances=30,
                              distribution=Gaussian(mean=0.0, sigma=1.0),
                              generator=sb.random.Generator(0))
    assert subs.shape == (4, 30)

    bcs = compute_persistent_homology(subs, max_dim=1)
    assert bcs.shape == (4, 30, 2)

    srs = barcode_to_stable_rank(bcs)
    rel = mean(srs, dim=1)
    assert rel.shape == (4, 2)
