import warnings

import numpy as np
import pytest

import stablebear as sb
from stablebear.persistence import barcode_to_stable_rank, compute_persistent_homology
from stablebear.reductions import mean
from stablebear.sampling import Gaussian, Uniform, subsample_relative


@pytest.fixture(params=[(np.float32, sb.pcloud32), (np.float64, sb.pcloud64)],
                ids=["f32", "f64"])
def float_kind(request):
    return request.param


def _ref_index_map(R):
    return {tuple(np.round(r, 5)): i for i, r in enumerate(R)}


def _sampled_indices(subs_elem, idx_map):
    return [idx_map[tuple(np.round(p, 5))] for p in np.asarray(subs_elem)]


def test_output_shape_and_dtype(float_kind):
    np_float, pcloud_dtype = float_kind
    R = np.random.default_rng(0).standard_normal((50, 3)).astype(np_float)
    X = np.random.default_rng(1).standard_normal((4, 3)).astype(np_float)

    subs = subsample_relative(R, X, sample_size=12, n_instances=7,
                     generator=sb.random.Generator(0))

    assert isinstance(subs, sb.PointCloudTensor)
    assert subs.dtype == pcloud_dtype
    assert subs.shape == (4, 7)
    assert subs[0, 0].shape == (12, 3)
    assert subs[3, 6].shape == (12, 3)


def test_gaussian_concentrates_on_nearest():
    # A small-sigma Gaussian over distance should sample the nearest reference
    # point far more than any other.
    R = (np.arange(30, dtype=np.float64)).reshape(-1, 1)
    X = np.array([[7.0]])  # exactly reference point index 7

    subs = subsample_relative(R, X, sample_size=20, n_instances=200,
                     distribution=Gaussian(mean=0.0, sigma=0.3),
                     generator=sb.random.Generator(0))

    idx_map = _ref_index_map(R)
    counts = np.zeros(len(R), dtype=int)
    for j in range(subs.shape[1]):
        for i in _sampled_indices(subs[0, j], idx_map):
            counts[i] += 1

    assert counts.argmax() == 7


def test_per_query_point_probabilities_differ():
    R = (np.arange(30, dtype=np.float64)).reshape(-1, 1)
    X = np.array([[3.0], [25.0]])

    subs = subsample_relative(R, X, sample_size=20, n_instances=100,
                     distribution=Gaussian(mean=0.0, sigma=1.0),
                     generator=sb.random.Generator(0))

    idx_map = _ref_index_map(R)
    means = []
    for q in range(2):
        idxs = []
        for j in range(subs.shape[1]):
            idxs.extend(_sampled_indices(subs[q, j], idx_map))
        means.append(np.mean(idxs))

    # The query near 3 should pull from small indices, the one near 25 from large.
    assert means[0] < means[1]


def test_reproducible_with_seed(float_kind):
    np_float, _ = float_kind
    R = np.random.default_rng(2).standard_normal((40, 2)).astype(np_float)
    X = np.random.default_rng(3).standard_normal((3, 2)).astype(np_float)

    a = subsample_relative(R, X, sample_size=10, n_instances=5, generator=sb.random.Generator(7))
    b = subsample_relative(R, X, sample_size=10, n_instances=5, generator=sb.random.Generator(7))
    c = subsample_relative(R, X, sample_size=10, n_instances=5, generator=sb.random.Generator(8))

    assert a.array_equal(b)
    assert not a.array_equal(c)


def test_consecutive_calls_advance_the_generator():
    # A draw must consume its seed block from the caller's generator, not a
    # private copy: two consecutive calls with the same generator draw fresh
    # samples, and reseeding replays the whole sequence call by call.
    R = np.random.default_rng(2).standard_normal((40, 2))
    X = np.random.default_rng(3).standard_normal((3, 2))

    gen = sb.random.Generator(7)
    a = subsample_relative(R, X, sample_size=10, n_instances=5, generator=gen)
    b = subsample_relative(R, X, sample_size=10, n_instances=5, generator=gen)
    assert not a.array_equal(b)

    gen2 = sb.random.Generator(7)
    a2 = subsample_relative(R, X, sample_size=10, n_instances=5, generator=gen2)
    b2 = subsample_relative(R, X, sample_size=10, n_instances=5, generator=gen2)
    assert a.array_equal(a2)
    assert b.array_equal(b2)


def test_consecutive_calls_advance_the_global_generator():
    # Same contract for generator=None: the global generator advances.
    R = np.random.default_rng(2).standard_normal((40, 2))
    X = np.random.default_rng(3).standard_normal((3, 2))

    sb.random.seed(11)
    a = subsample_relative(R, X, sample_size=10, n_instances=5)
    b = subsample_relative(R, X, sample_size=10, n_instances=5)
    assert not a.array_equal(b)

    sb.random.seed(11)
    a2 = subsample_relative(R, X, sample_size=10, n_instances=5)
    b2 = subsample_relative(R, X, sample_size=10, n_instances=5)
    assert a.array_equal(a2)
    assert b.array_equal(b2)


def test_verbose_matches_nonverbose():
    # verbose=True drives the result through the stoppable-task progress loop;
    # it must produce exactly the same (deterministic) subsamples as verbose=False.
    R = np.random.default_rng(6).standard_normal((40, 2))
    X = np.random.default_rng(7).standard_normal((3, 2))

    quiet = subsample_relative(R, X, sample_size=10, n_instances=5,
                      generator=sb.random.Generator(11), verbose=False)
    loud = subsample_relative(R, X, sample_size=10, n_instances=5,
                     generator=sb.random.Generator(11), verbose=True)

    assert isinstance(loud, sb.PointCloudTensor)
    assert loud.array_equal(quiet)


def test_uniform_samples_all_reference_points():
    # With equal weights every reference point should eventually be drawn.
    R = (np.arange(20, dtype=np.float64)).reshape(-1, 1)
    X = np.array([[100.0]])  # far from R: a distance-based distribution would skew

    subs = subsample_relative(R, X, sample_size=5, n_instances=400,
                     distribution=Uniform(), generator=sb.random.Generator(0))

    idx_map = _ref_index_map(R)
    seen = set()
    for j in range(subs.shape[1]):
        seen.update(_sampled_indices(subs[0, j], idx_map))
    assert seen == set(range(20))


def test_uniform_disk_samples_only_within_radius():
    # Uniform(high=r) is a disk: only reference points within distance r of the
    # query may be drawn, and all of them should be (eventually).
    R = (np.arange(20, dtype=np.float64)).reshape(-1, 1)
    X = np.array([[7.0]])  # reference index 7
    radius = 3.0

    subs = subsample_relative(R, X, sample_size=5, n_instances=400,
                     distribution=Uniform(high=radius),
                     generator=sb.random.Generator(0))

    idx_map = _ref_index_map(R)
    seen = set()
    for j in range(subs.shape[1]):
        seen.update(_sampled_indices(subs[0, j], idx_map))

    assert seen == {i for i in range(20) if abs(R[i, 0] - 7.0) <= radius}


def test_uniform_annulus_samples_only_within_band():
    # Uniform(low, high) is a ring: only points whose distance to the query
    # falls in [low, high] may be drawn.
    R = (np.arange(30, dtype=np.float64)).reshape(-1, 1)
    X = np.array([[15.0]])  # reference index 15
    low, high = 4.0, 8.0

    subs = subsample_relative(R, X, sample_size=5, n_instances=500,
                     distribution=Uniform(low=low, high=high),
                     generator=sb.random.Generator(0))

    idx_map = _ref_index_map(R)
    seen = set()
    for j in range(subs.shape[1]):
        seen.update(_sampled_indices(subs[0, j], idx_map))

    assert seen == {i for i in range(30) if low <= abs(R[i, 0] - 15.0) <= high}


@pytest.mark.parametrize("kwargs", [{"low": -1.0}, {"high": 0.0},
                                    {"low": 2.0, "high": 1.0},
                                    {"low": 1.0, "high": 1.0}])
def test_uniform_invalid_radii_raise(kwargs):
    with pytest.raises(ValueError):
        Uniform(**kwargs)


@pytest.mark.parametrize("args,low,high", [
    ((3.0,), 3.0, float("inf")),
    ((1.0, 3.0), 1.0, 3.0),
], ids=["one", "two"])
def test_uniform_accepts_positional_args(args, low, high):
    # Positional band edges bind in (low, high) order.
    spec = Uniform(*args)
    assert spec.low == low
    assert spec.high == high


@pytest.mark.parametrize("args,mean,sigma", [
    ((0.3,), 0.3, 1.0),
    ((0.0, 0.3), 0.0, 0.3),
], ids=["one", "two"])
def test_gaussian_accepts_positional_args(args, mean, sigma):
    # Positional parameters bind in (mean, sigma) order.
    spec = Gaussian(*args)
    assert spec.mean == mean
    assert spec.sigma == sigma


@pytest.mark.parametrize("kwargs", [
    {"sigma": 0.0}, {"sigma": -1.0}, {"sigma": float("nan")}, {"sigma": float("inf")},
    {"mean": float("nan")}, {"mean": float("inf")}, {"mean": float("-inf")},
])
def test_gaussian_invalid_params_raise(kwargs):
    # Non-finite parameters would not error downstream: mean=nan/inf silently
    # empties every subsample, sigma=inf silently degenerates to uniform
    # sampling. They must be rejected at construction, like Uniform's.
    with pytest.raises(ValueError):
        Gaussian(**kwargs)


def test_nan_reference_coordinate_propagates_from_worker():
    # Weight-row validation runs inside the parallel walk, not in the task's
    # synchronous prologue, so this is the one path that exercises a worker
    # exception reaching the caller. A NaN coordinate survives the log-space
    # max-shift (max() drops NaNs) and poisons the row sum, which
    # prepare_weight_row rejects on whichever worker thread drew that query.
    R = np.zeros((8, 2))
    R[3, 0] = np.nan

    with pytest.raises(ValueError, match="positive sum"):
        subsample_relative(sb.FloatTensor(R), R[:1], sample_size=3, n_instances=2,
                           distribution=Gaussian(mean=1.0, sigma=0.5))


def test_without_replacement_gives_distinct_points():
    R = (np.arange(15, dtype=np.float64)).reshape(-1, 1)
    X = np.array([[0.0]])

    subs = subsample_relative(R, X, sample_size=10, n_instances=20,
                     distribution=Uniform(), replace=False,
                     generator=sb.random.Generator(0))

    idx_map = _ref_index_map(R)
    for j in range(subs.shape[1]):
        idxs = _sampled_indices(subs[0, j], idx_map)
        assert len(idxs) == len(set(idxs))


def test_without_replacement_larger_than_reference_draws_all_points():
    # sample_size is a maximum: without replacement, asking for more points
    # than the reference holds yields every reference point once.
    R = np.random.default_rng(0).standard_normal((5, 2))
    X = np.zeros((1, 2))

    subs = subsample_relative(R, X, sample_size=6, n_instances=4, replace=False,
                              generator=sb.random.Generator(0))

    for j in range(subs.shape[1]):
        idx = np.asarray(subs[0, j].indices)
        assert sorted(idx) == list(range(5))


def test_without_replacement_small_region_gives_ragged_subsamples():
    # A Uniform disk holding fewer points than sample_size: each subsample
    # shrinks to exactly the eligible points (ragged, 0 < size < sample_size).
    R = np.arange(20, dtype=np.float64).reshape(-1, 1)
    X = np.array([[7.0]])
    eligible = {6, 7, 8}  # |R[i] - 7| <= 1.5

    subs = subsample_relative(R, X, sample_size=10, n_instances=15,
                              distribution=Uniform(high=1.5), replace=False,
                              generator=sb.random.Generator(0))

    for j in range(subs.shape[1]):
        idx = np.asarray(subs[0, j].indices)
        assert set(int(i) for i in idx) == eligible
        assert subs[0, j].shape == (3, 1)


def test_with_replacement_small_region_still_fills_sample_size():
    # With replacement, a single eligible point suffices to fill sample_size.
    R = np.arange(20, dtype=np.float64).reshape(-1, 1)
    X = np.array([[7.0]])

    subs = subsample_relative(R, X, sample_size=10, n_instances=5,
                              distribution=Uniform(high=0.5), replace=True,
                              generator=sb.random.Generator(0))

    for j in range(subs.shape[1]):
        idx = np.asarray(subs[0, j].indices)
        assert idx.shape == (10,)
        assert set(int(i) for i in idx) == {7}


@pytest.mark.parametrize("replace", [True, False], ids=["replace", "no-replace"])
def test_empty_region_gives_empty_subsamples_and_warns_when_verbose(replace):
    # A validly-specified region that no reference point falls in: the query's
    # subsamples are length-0 indexed views, and with verbose=True a warning
    # names the affected query.
    R = np.arange(20, dtype=np.float64).reshape(-1, 1)
    X = np.array([[7.0], [100.0]])  # second query's disk is empty

    with pytest.warns(UserWarning, match="1 query point.*query indices: 1"):
        subs = subsample_relative(R, X, sample_size=5, n_instances=3,
                                  distribution=Uniform(high=2.0), replace=replace,
                                  generator=sb.random.Generator(0), verbose=True)

    for j in range(subs.shape[1]):
        assert np.asarray(subs[0, j].indices).size > 0
        el = subs[1, j]
        assert el.is_indexed
        assert np.asarray(el.indices).size == 0


def test_empty_region_is_silent_without_verbose():
    # Quiet runs (the default) get the same empty subsamples with no warning.
    R = np.arange(20, dtype=np.float64).reshape(-1, 1)
    X = np.array([[100.0]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        subs = subsample_relative(R, X, sample_size=5, n_instances=3,
                                  distribution=Uniform(high=2.0),
                                  generator=sb.random.Generator(0))
    assert not caught
    assert np.asarray(subs[0, 0].indices).size == 0

def test_gaussian_far_query_samples_nearest_point_not_empty():
    # exp(-0.5 (d/sigma)^2) underflows to exactly 0 for huge d/sigma, but the
    # Gaussian weights are computed in log space with a per-query max shift,
    # so a far query still samples its exact Gaussian — concentrated on the
    # nearest reference point — instead of coming back empty with a warning.
    R = np.arange(20, dtype=np.float64).reshape(-1, 1)
    X = np.array([[1e6]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        subs = subsample_relative(R, X, sample_size=5, n_instances=2,
                                  distribution=Gaussian(mean=0.0, sigma=1.0),
                                  generator=sb.random.Generator(0), verbose=True)
    assert not [w for w in caught if issubclass(w.category, UserWarning)]

    for j in range(2):
        idx = np.asarray(subs[0, j].indices)
        assert idx.shape == (5,)
        assert set(int(i) for i in idx) == {19}  # the nearest reference point


def test_dimension_mismatch_raises():
    R = np.zeros((10, 3))
    X = np.zeros((2, 2))
    with pytest.raises(ValueError):
        subsample_relative(R, X, sample_size=3, n_instances=1)


@pytest.mark.parametrize("bad", [lambda v: np.asarray(v), object(), 3.0])
def test_custom_distribution_rejected(bad):
    # Only the built-in specs (Gaussian, Uniform) are accepted; arbitrary
    # callables and other objects must be rejected.
    R = np.zeros((10, 2))
    X = np.zeros((1, 2))
    with pytest.raises(ValueError):
        subsample_relative(R, X, sample_size=3, n_instances=1, distribution=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_counts_raise(bad):
    R = np.zeros((10, 2))
    X = np.zeros((1, 2))
    with pytest.raises(ValueError):
        subsample_relative(R, X, sample_size=bad, n_instances=1)
    with pytest.raises(ValueError):
        subsample_relative(R, X, sample_size=3, n_instances=bad)


def test_subsamples_are_indexed_views():
    R = np.random.default_rng(0).standard_normal((100, 8))
    X = np.random.default_rng(1).standard_normal((3, 8))

    subs = subsample_relative(R, X, sample_size=10, n_instances=5,
                     distribution=Gaussian(mean=0.0, sigma=1.0), generator=sb.random.Generator(0))

    el = subs[0, 0]
    assert el.is_indexed
    assert el.shape == (10, 8)

    idx = np.asarray(el.indices)
    assert idx.shape == (10,)
    # The materialized coordinates are exactly the referenced source rows
    # (no coordinates are copied until materialization).
    assert np.array_equal(el.to_numpy(), R[idx])


def test_indexed_subsample_write_detaches():
    # Writing to an indexed view is copy-on-write: the view detaches from the
    # shared source (materializing a private copy), so the write succeeds, the
    # cloud is no longer indexed, and the reference cloud is untouched.
    R = np.random.default_rng(0).standard_normal((40, 3))
    X = np.random.default_rng(1).standard_normal((2, 3))

    subs = subsample_relative(R, X, sample_size=8, n_instances=3,
                              generator=sb.random.Generator(0))
    el = subs[0, 0]
    assert el.is_indexed

    before = R.copy()
    el[0, 0] = 123.0
    assert float(el[0, 0]) == 123.0
    assert not el.is_indexed
    assert np.array_equal(R, before)


def test_owning_pointcloud_is_mutable():
    # A cloud that owns its coordinates (not an indexed view) is mutable, and the
    # write lands on the stored cell.
    t = sb.zeros((1,), dtype=sb.pcloud64)
    t[0] = sb.FloatTensor(np.zeros((3, 2)))
    el = t[0]
    assert not el.is_indexed

    el[0, 0] = 7.0
    assert float(t[0][0, 0]) == 7.0


def test_query_by_index_matches_coordinates():
    # Selecting query points by their order in the reference must match passing
    # those reference coordinates directly.
    R = np.random.default_rng(0).standard_normal((50, 3))
    idx = np.array([1, 3, 10, 25])

    by_index = subsample_relative(R, idx, sample_size=8, n_instances=5,
                                  distribution=Gaussian(mean=0.0, sigma=1.0),
                                  generator=sb.random.Generator(0))
    by_coords = subsample_relative(R, R[idx], sample_size=8, n_instances=5,
                                   distribution=Gaussian(mean=0.0, sigma=1.0),
                                   generator=sb.random.Generator(0))

    assert by_index.shape == (4, 5)
    assert by_index.array_equal(by_coords)


def test_negative_query_indices_match_positive():
    # Negative indices count from the end, as in NumPy.
    R = np.random.default_rng(0).standard_normal((30, 2))

    negative = subsample_relative(R, np.array([-1, -30]), sample_size=6, n_instances=4,
                                  generator=sb.random.Generator(3))
    positive = subsample_relative(R, np.array([29, 0]), sample_size=6, n_instances=4,
                                  generator=sb.random.Generator(3))
    assert negative.array_equal(positive)


def test_query_none_uses_all_reference_points():
    # The default query is the reference cloud itself: one output row per
    # reference point.
    R = np.random.default_rng(0).standard_normal((17, 2))
    subs = subsample_relative(R, sample_size=4, n_instances=3,
                              generator=sb.random.Generator(0))
    assert subs.shape == (17, 3)


def test_query_index_out_of_range_raises():
    R = np.zeros((10, 2))
    with pytest.raises(ValueError):
        subsample_relative(R, np.array([10]), sample_size=3, n_instances=1)
    with pytest.raises(ValueError):
        subsample_relative(R, np.array([-11]), sample_size=3, n_instances=1)


def test_huge_unsigned_query_index_raises():
    # A uint64 value >= 2^63 casts to a negative int64; only *signed* input
    # gets the negative-index treatment, so it must fail the range check
    # rather than wrap to a valid row.
    R = np.zeros((10, 2))
    with pytest.raises(ValueError):
        subsample_relative(R, np.array([2**63], dtype=np.uint64),
                           sample_size=3, n_instances=1)


@pytest.mark.parametrize("shape", [(10,), (2, 5, 3)], ids=["1d", "3d"])
def test_non_2d_reference_raises(shape):
    with pytest.raises(ValueError):
        subsample_relative(np.zeros(shape), sample_size=3, n_instances=1)


def test_non_2d_coordinate_query_raises():
    R = np.zeros((10, 3))
    with pytest.raises(ValueError):
        subsample_relative(R, np.zeros((2, 2, 3)), sample_size=3, n_instances=1)


@pytest.mark.parametrize("n", [1, 2, 12], ids=["1x1", "2x2", "12x12"])
def test_square_reference_rejected_as_ambiguous(n):
    R = np.random.default_rng(0).random((n, n))
    with pytest.raises(ValueError, match="ambiguous"):
        subsample_relative(R, np.array([0]), sample_size=3, n_instances=2)


def test_float_tensor_wrapper_accepts_square_reference():
    # An explicit FloatTensor declares "rows are points" and skips the guard.
    R = sb.FloatTensor(np.random.default_rng(0).random((12, 12)))
    subs = subsample_relative(R, np.array([0]), sample_size=3, n_instances=2,
                              generator=sb.random.Generator(0))
    assert isinstance(subs, sb.PointCloudTensor)
    assert subs.shape == (1, 2)
    assert subs[0, 0].shape == (3, 12)


def test_distance_matrix_wrapper_accepts_square_reference():
    A = np.random.default_rng(0).random((12, 12))
    D = (A + A.T) / 2
    np.fill_diagonal(D, 0.0)
    subs = subsample_relative(sb.DistanceMatrix.from_dense(D), np.array([0]),
                              sample_size=3, n_instances=2,
                              generator=sb.random.Generator(0))
    assert isinstance(subs, sb.DistanceMatrixTensor)


def test_non_square_reference_needs_no_wrapper():
    R = np.random.default_rng(0).random((12, 3))
    subs = subsample_relative(R, np.array([0]), sample_size=3, n_instances=2,
                              generator=sb.random.Generator(0))
    assert isinstance(subs, sb.PointCloudTensor)


@pytest.mark.parametrize("wrap", [
    lambda R: sb.PointCloudTensor(R[None])[0],   # a single PointCloud
    lambda R: sb.PointCloudTensor(R),            # 0-d single-cloud tensor
    lambda R: sb.PointCloudTensor(R[None]),      # rank-1 single-cloud tensor
], ids=["pointcloud", "0d-tensor", "1d-tensor"])
def test_point_cloud_reference_matches_array(float_kind, wrap):
    # A PointCloud (or a single-cloud PointCloudTensor) reference is taken as a
    # point cloud, giving the same draw as passing the coordinates directly.
    np_float, pcloud_dtype = float_kind
    R = np.random.default_rng(0).standard_normal((40, 3)).astype(np_float)
    X = np.random.default_rng(1).standard_normal((4, 3)).astype(np_float)

    from_cloud = subsample_relative(wrap(R), X, sample_size=8, n_instances=5,
                                    generator=sb.random.Generator(0))
    from_array = subsample_relative(sb.FloatTensor(R), X, sample_size=8, n_instances=5,
                                    generator=sb.random.Generator(0))
    assert isinstance(from_cloud, sb.PointCloudTensor)
    assert from_cloud.dtype == pcloud_dtype
    assert from_cloud.array_equal(from_array)


def test_subsample_can_be_fed_back_as_reference():
    # The motivating workflow: a subsample -- an indexed-view PointCloud -- is a
    # valid reference for a further subsampling pass, without materializing it.
    R = np.random.default_rng(0).standard_normal((60, 3))
    first = subsample_relative(R, sample_size=20, n_instances=1,
                               generator=sb.random.Generator(0))
    sub = first[0, 0]
    assert sub.is_indexed  # shares R's coordinates through an index array

    second = subsample_relative(sub, sample_size=5, n_instances=4,
                                generator=sb.random.Generator(1))
    assert isinstance(second, sb.PointCloudTensor)
    assert second.shape == (20, 4)  # query=None: one row per point of `sub`

    # Identical to materializing the subsample's coordinates first.
    expected = subsample_relative(sb.FloatTensor(sub.to_numpy()),
                                  sample_size=5, n_instances=4,
                                  generator=sb.random.Generator(1))
    assert second.array_equal(expected)


def test_multi_cloud_tensor_reference_rejected():
    # A PointCloudTensor holding more than one cloud is ambiguous as a single
    # reference cloud and must be rejected with a clear error.
    pct = sb.PointCloudTensor(np.random.default_rng(0).standard_normal((3, 10, 2)))
    assert pct.shape == (3,)
    with pytest.raises(ValueError, match="single reference cloud"):
        subsample_relative(pct, np.array([0]), sample_size=3, n_instances=2)


def test_point_cloud_reference_accepts_square():
    # A PointCloud is unambiguously points, so a square (n, n) cloud skips the
    # square-array ambiguity guard that rejects a raw ndarray.
    R = np.random.default_rng(0).random((12, 12))
    pc = sb.PointCloudTensor(R[None])[0]
    subs = subsample_relative(pc, np.array([0]), sample_size=3, n_instances=2,
                              generator=sb.random.Generator(0))
    assert isinstance(subs, sb.PointCloudTensor)
    assert subs[0, 0].shape == (3, 12)


def test_pipeline_to_relative_stable_rank():
    R = np.random.default_rng(0).standard_normal((200, 2))
    X = np.random.default_rng(1).standard_normal((4, 2))

    subs = subsample_relative(R, X, sample_size=25, n_instances=30,
                     distribution=Gaussian(mean=0.0, sigma=1.0),
                     generator=sb.random.Generator(0))

    bcs = compute_persistent_homology(subs, max_dim=1)
    assert bcs.shape == (4, 30, 2)

    srs = barcode_to_stable_rank(bcs)
    rel = mean(srs, dim=1)
    assert rel.shape == (4, 2)
