#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
from scipy.spatial.distance import pdist, squareform

import masspcf as mpcf
import masspcf.persistence as mpers


def _points_to_distmat(points, dtype):
    """Build a DistanceMatrix from a NumPy point array."""
    dists = squareform(pdist(points))
    n = dists.shape[0]
    dm = mpcf.DistanceMatrix(n, dtype=dtype)
    for i in range(n):
        for j in range(i):
            dm[i, j] = dists[i, j]
    return dm


def test_distmat_single_matches_point_cloud():
    """Single DistanceMatrix gives the same barcodes as the equivalent point cloud."""
    points = np.array([
        [0.0, 0.0],
        [0.0, 4.0],
        [3.0, 0.0],
        [3.0, 4.0],
    ])

    bcs_pc = mpers.compute_persistent_homology(points, maxDim=2, verbose=False)

    dm = _points_to_distmat(points, mpcf.f64)
    bcs_dm = mpers.compute_persistent_homology(dm, maxDim=2, verbose=False)

    assert isinstance(bcs_dm, mpers.Barcode64Tensor)
    assert bcs_dm.shape == bcs_pc.shape

    for k in range(bcs_pc.shape[0]):
        assert bcs_dm[k].is_isomorphic_to(bcs_pc[k])


def test_distmat_single_f32():
    """DistanceMatrix with f32 dtype produces Barcode32Tensor."""
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    dm = _points_to_distmat(points, mpcf.f32)
    bcs = mpers.compute_persistent_homology(dm, maxDim=1, verbose=False)

    assert isinstance(bcs, mpers.Barcode32Tensor)
    assert bcs.shape == (2,)


def test_distmat_tensor_matches_point_cloud():
    """DistanceMatrixTensor gives the same barcodes as PointCloudTensor."""
    np.random.seed(42)

    pclouds = mpcf.zeros((3,), dtype=mpcf.pcloud64)
    dmats = mpcf.zeros((3,), dtype=mpcf.distmat64)

    for i in range(3):
        pts = np.random.randn(8, 3)
        pclouds[i] = pts
        dmats[i] = _points_to_distmat(pts, mpcf.f64)

    bcs_pc = mpers.compute_persistent_homology(pclouds, maxDim=1, verbose=False)
    bcs_dm = mpers.compute_persistent_homology(dmats, maxDim=1, verbose=False)

    assert isinstance(bcs_dm, mpers.Barcode64Tensor)
    assert bcs_dm.shape == bcs_pc.shape

    for i in range(3):
        for k in range(2):
            assert bcs_dm[i, k].is_isomorphic_to(bcs_pc[i, k])


def test_distmat_reduced_homology():
    """Reduced homology with DistanceMatrix omits the essential H0 bar."""
    points = np.array([
        [0.0, 0.0],
        [0.0, 4.0],
        [3.0, 0.0],
        [3.0, 4.0],
    ])

    dm = _points_to_distmat(points, mpcf.f64)

    bcs_unreduced = mpers.compute_persistent_homology(dm, maxDim=1, reduced=False, verbose=False)
    bcs_reduced = mpers.compute_persistent_homology(dm, maxDim=1, reduced=True, verbose=False)

    # Unreduced H0 should have one more bar (the essential [0, inf))
    assert len(bcs_unreduced[0]) == len(bcs_reduced[0]) + 1

    # H1 should be the same
    assert bcs_unreduced[1].is_isomorphic_to(bcs_reduced[1])
