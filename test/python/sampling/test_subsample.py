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

    for i in range(3):
        for j in range(5):
            assert np.array_equal(np.asarray(a[i, j]), np.asarray(b[i, j]))

    any_diff = any(
        not np.array_equal(np.asarray(a[i, j]), np.asarray(c[i, j]))
        for i in range(3) for j in range(5)
    )
    assert any_diff


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
    assert loud.shape == quiet.shape
    for i in range(3):
        for j in range(5):
            assert np.array_equal(np.asarray(loud[i, j]), np.asarray(quiet[i, j]))


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


@pytest.mark.parametrize("args", [(3.0,), (1.0, 3.0)], ids=["one", "two"])
def test_uniform_positional_args_rejected(args):
    # The band edges are keyword-only: a bare Uniform(3.0) is ambiguous (disk
    # of radius 3 vs everything beyond 3), so it must not be accepted.
    with pytest.raises(TypeError):
        Uniform(*args)


@pytest.mark.parametrize("args", [(0.3,), (0.0, 0.3)], ids=["one", "two"])
def test_gaussian_positional_args_rejected(args):
    # mean and sigma are keyword-only: a bare Gaussian(0.3) is ambiguous
    # (mean or sigma?), so it must not be accepted.
    with pytest.raises(TypeError):
        Gaussian(*args)


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


def test_gaussian_underflow_region_is_empty_not_error():
    # exp(-0.5 (d/sigma)^2) underflows to exactly 0 for huge d/sigma, so even a
    # Gaussian can leave a query with an all-zero weight row. That must behave
    # like any other empty region (empty subsamples + warning), not crash.
    R = np.arange(20, dtype=np.float64).reshape(-1, 1)
    X = np.array([[1e6]])

    with pytest.warns(UserWarning):
        subs = subsample_relative(R, X, sample_size=5, n_instances=2,
                                  distribution=Gaussian(mean=0.0, sigma=1.0),
                                  generator=sb.random.Generator(0), verbose=True)
    assert np.asarray(subs[0, 0].indices).size == 0


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


def test_indexed_subsample_is_read_only():
    # An indexed view shares the reference cloud's coordinates, so writing to it
    # would corrupt the source; it must reject in-place assignment. materialize()
    # gives an independent, writable copy.
    R = np.random.default_rng(0).standard_normal((40, 3))
    X = np.random.default_rng(1).standard_normal((2, 3))

    subs = subsample_relative(R, X, sample_size=8, n_instances=3,
                              generator=sb.random.Generator(0))
    el = subs[0, 0]
    assert el.is_indexed

    with pytest.raises(TypeError):
        el[0, 0] = 1.0

    # A writable copy: mutating it leaves the source reference cloud untouched.
    m = el.materialize()
    m[0, 0] = 123.0
    assert float(m[0, 0]) == 123.0
    assert not np.any(R == 123.0)


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
    for i in range(4):
        for j in range(5):
            assert np.array_equal(np.asarray(by_index[i, j]), np.asarray(by_coords[i, j]))


def test_query_index_out_of_range_raises():
    R = np.zeros((10, 2))
    with pytest.raises(ValueError):
        subsample_relative(R, np.array([10]), sample_size=3, n_instances=1)


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
