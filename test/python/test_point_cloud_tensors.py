import numpy as np
import pytest

import stablebear as sb


def test_can_create_point_clouds():
    X = sb.zeros((2,), dtype=sb.pcloud64)

    assert isinstance(X, sb.PointCloudTensor)
    assert X.dtype == sb.pcloud64

    X[0] = np.random.randn(10, 2)
    X[1] = np.random.randn(20, 2)

    assert X[0].shape == (10, 2)
    assert X[1].shape == (20, 2)

    Y = sb.zeros((2, 3), dtype=sb.pcloud32)

    assert isinstance(Y, sb.PointCloudTensor)
    assert Y.dtype == sb.pcloud32

    Y[0, 0] = np.random.randn(30, 2)
    Y[1, 1] = np.random.randn(40, 15)

    assert Y[0, 0].shape == (30, 2)
    assert Y[1, 1].shape == (40, 15)


def test_point_clouds_must_be_rank_2():
    X = sb.zeros((2,), dtype=sb.pcloud64)
    with pytest.raises(ValueError):
        X[0] = np.random.randn(30, 2, 20)


def test_single_cloud_is_subscriptable():
    # A 0-d PointCloudTensor wraps a single cloud; it should be indexable as
    # its (n_points, dim) array so the natural pc[:, 0] / pc[:, 1] plotting
    # idiom works directly (see issue #133).
    arr = np.random.RandomState(0).rand(6, 2)
    pc = sb.PointCloudTensor(arr)
    assert pc.ndim == 0

    assert pc[:, 0].array_equal(arr[:, 0])
    assert pc[:, 1].array_equal(arr[:, 1])
    assert pc[0].array_equal(arr[0])
    assert pc[1:3].array_equal(arr[1:3])

    # Whole-cloud element access is unchanged.
    assert pc[()].array_equal(arr)
    assert pc[...].array_equal(arr)


def test_point_selection_returns_a_shared_view():
    # Selecting whole points is the indexed-view machinery: the result shares
    # the source coordinates instead of copying them.
    arr = np.random.RandomState(3).rand(10, 2)
    T = sb.zeros((1,), dtype=sb.pcloud64)
    T[0] = arr
    pc = T[0]

    for key, expected in [(slice(1, 4), arr[1:4]),
                          ([3, 7], arr[[3, 7]]),
                          ([-1, -2], arr[[-1, -2]]),
                          (np.arange(10) % 3 == 0, arr[np.arange(10) % 3 == 0])]:
        selected = pc[key]
        assert isinstance(selected, sb.PointCloud)
        assert selected.is_indexed
        assert selected.array_equal(expected)


def test_point_selection_of_a_view_composes_indices():
    # A view's indices address the source coordinates, so slicing a view must
    # compose through them rather than index the view's own rows.
    arr = np.random.RandomState(4).rand(12, 3)
    T = sb.zeros((1,), dtype=sb.pcloud64)
    T[0] = arr
    pc = T[0]

    view = pc[2:9]
    nested = view[1:3]
    assert nested.is_indexed
    assert nested.array_equal(arr[2:9][1:3])
    # ...and the original view is untouched by the nested selection.
    assert view.array_equal(arr[2:9])


def test_coordinate_indexing_still_materializes():
    # Anything that drops the point axis or reaches the coordinate axis yields
    # numbers, so the plotting idiom keeps working.
    arr = np.random.RandomState(5).rand(8, 2)
    T = sb.zeros((1,), dtype=sb.pcloud64)
    T[0] = arr
    pc = T[0]

    for key, expected in [(0, arr[0]), ((slice(None), 0), arr[:, 0]),
                          ((slice(1, 4), slice(0, 1)), arr[1:4, 0:1]),
                          (Ellipsis, arr), ((), arr)]:
        selected = pc[key]
        assert isinstance(selected, sb.FloatTensor)
        assert selected.array_equal(expected)


def test_out_of_range_point_selection_raises():
    arr = np.random.RandomState(6).rand(5, 2)
    T = sb.zeros((1,), dtype=sb.pcloud64)
    T[0] = arr

    with pytest.raises(IndexError):
        T[0][[0, 5]]
    with pytest.raises(IndexError):
        T[0][np.array([True, False])]  # mask shorter than the cloud


def test_storing_a_view_keeps_it_indexed():
    # A PointCloud is the tensor's element type, so storing one must not
    # materialize it -- the stored cell keeps sharing the source coordinates.
    arr = np.random.RandomState(7).rand(10, 2)
    T = sb.zeros((1,), dtype=sb.pcloud64)
    T[0] = arr

    dest = sb.zeros((2,), dtype=sb.pcloud64)
    dest[0] = T[0][2:6]

    assert dest[0].is_indexed
    assert dest[0].array_equal(arr[2:6])


def test_storing_a_view_does_not_alias_the_source_cloud():
    # The store copies the index array (store_copy -> PointCloud::copy), so
    # writing to the cloud the view was taken from leaves the stored cell alone.
    arr = np.random.RandomState(8).rand(6, 2)
    T = sb.zeros((1,), dtype=sb.pcloud64)
    T[0] = arr

    dest = sb.zeros((1,), dtype=sb.pcloud64)
    dest[0] = T[0][1:4]
    T[0][0, 0] = 99.0

    assert dest[0].array_equal(arr[1:4])


def test_storing_a_cloud_of_another_precision_converts():
    arr = np.random.RandomState(9).rand(5, 2)
    T = sb.zeros((1,), dtype=sb.pcloud32)
    T[0] = arr

    dest = sb.zeros((1,), dtype=sb.pcloud64)
    dest[0] = T[0][1:4]

    assert not dest[0].is_indexed  # converted through its coordinates
    assert np.allclose(dest[0].to_numpy(), arr[1:4])


def test_tensor_and_point_cloud_share_the_indexing_interface():
    # bwehlin's Indexable ask: one contract, both containers.
    from stablebear._indexable import Indexable

    assert issubclass(sb.FloatTensor, Indexable)
    assert issubclass(sb.PointCloud, Indexable)


def test_tensor_of_clouds_indexing_unchanged():
    # Rank >= 1 tensors still index over clouds, not into them: T[0] is one
    # cloud (a PointCloud), which is itself subscriptable as its (n_points, dim)
    # coordinates.
    arr = np.random.RandomState(1).rand(5, 2)
    T = sb.zeros((3,), dtype=sb.pcloud64)
    T[0] = arr

    assert isinstance(T[0], sb.PointCloud)
    assert T[0].shape == (5, 2)
    assert T[0][:, 1].array_equal(arr[:, 1])

    sub = T[1:]
    assert isinstance(sub, sb.PointCloudTensor)
    assert sub.shape == (2,)

    # Selecting one cloud from a higher-rank tensor, then column-indexing it.
    grid = sb.zeros((2, 3), dtype=sb.pcloud64)
    grid[0, 1] = arr
    assert grid[0, 1][:, 0].array_equal(arr[:, 0])
    assert grid[0, 1][3].array_equal(arr[3])


def test_stored_is_same_as_numpy():
    shape = (10, 20, 30)
    pclouds = sb.zeros(shape, dtype=sb.pcloud64)
    X = np.random.randn(10, 2).astype(np.float64)

    pclouds[0, 1, 2] = X
    assert pclouds[0, 1, 2].array_equal(X)

    pclouds = sb.zeros(shape, dtype=sb.pcloud32)
    X = np.random.randn(10, 2).astype(np.float32)

    pclouds[0, 1, 2] = X
    assert pclouds[0, 1, 2].array_equal(X)
