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
import pytest

import masspcf as mpcf
import masspcf.persistence as mpers
from masspcf.persistence.ripser import _ripser_plusplus_available


pytestmark = pytest.mark.skipif(
    not _ripser_plusplus_available(),
    reason="Ripser++ requires the CUDA backend",
)


# Loose enough to absorb 1-ulp differences between the two algorithms'
# summation orders in float32 while still catching real discrepancies.
_ATOL = 1e-5
_RTOL = 1e-5


def _rectangle_point_cloud():
    # Distance space is "two 3-4-5 triangles"; bars fall on integer values.
    X = np.zeros((4, 2))
    X[0, :] = [0.0, 0.0]
    X[1, :] = [0.0, 4.0]
    X[2, :] = [3.0, 0.0]
    X[3, :] = [3.0, 4.0]
    return X


def test_ripser_plusplus_unreduced_homology_exact_on_rectangle():
    X = _rectangle_point_cloud()

    bcs = mpers.compute_persistent_homology(
        X,
        max_dim=2,
        complex_type=mpers.ComplexType.VietorisRips,
        distance_type=mpers.DistanceType.Euclidean,
        device="gpu",
    )

    expected_h0 = mpers.Barcode(np.array([
        [0.0, np.inf], [0.0, 3.0], [0.0, 3.0], [0.0, 4.0],
    ]))
    expected_h1 = mpers.Barcode(np.array([[4.0, 5.0]]))
    expected_h2 = mpers.Barcode(np.zeros((0, 2)))

    assert expected_h0.is_isomorphic_to(bcs[0])
    assert expected_h1.is_isomorphic_to(bcs[1])
    assert expected_h2.is_isomorphic_to(bcs[2])


def test_ripser_plusplus_reduced_homology_exact_on_rectangle():
    X = _rectangle_point_cloud()

    bcs = mpers.compute_persistent_homology(
        X,
        max_dim=2,
        reduced=True,
        complex_type=mpers.ComplexType.VietorisRips,
        distance_type=mpers.DistanceType.Euclidean,
        device="gpu",
    )

    expected_h0 = mpers.Barcode(np.array([[0.0, 3.0], [0.0, 3.0], [0.0, 4.0]]))
    expected_h1 = mpers.Barcode(np.array([[4.0, 5.0]]))
    expected_h2 = mpers.Barcode(np.zeros((0, 2)))

    assert expected_h0.is_isomorphic_to(bcs[0])
    assert expected_h1.is_isomorphic_to(bcs[1])
    assert expected_h2.is_isomorphic_to(bcs[2])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ripser_plusplus_matches_ripser_on_random_pcloud(dtype):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3)).astype(dtype)

    bcs_cpu = mpers.compute_persistent_homology(X, max_dim=1, device="cpu")
    bcs_gpu = mpers.compute_persistent_homology(X, max_dim=1, device="gpu")

    assert bcs_cpu[0].is_isomorphic_to(bcs_gpu[0], atol=_ATOL, rtol=_RTOL)
    assert bcs_cpu[1].is_isomorphic_to(bcs_gpu[1], atol=_ATOL, rtol=_RTOL)


def test_ripser_plusplus_matches_ripser_on_circle():
    # Noisy circle -> one prominent H1 bar.
    rng = np.random.default_rng(1)
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    X = np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float32)
    X += 0.02 * rng.standard_normal(X.shape).astype(np.float32)

    bcs_cpu = mpers.compute_persistent_homology(X, max_dim=1, device="cpu")
    bcs_gpu = mpers.compute_persistent_homology(X, max_dim=1, device="gpu")

    assert bcs_cpu[0].is_isomorphic_to(bcs_gpu[0], atol=_ATOL, rtol=_RTOL)
    assert bcs_cpu[1].is_isomorphic_to(bcs_gpu[1], atol=_ATOL, rtol=_RTOL)


def test_ripser_plusplus_tensor_of_pclouds():
    rng = np.random.default_rng(42)
    X = mpcf.zeros((2, 3), dtype=mpcf.pcloud32)
    for i in range(2):
        for j in range(3):
            X[i, j] = rng.standard_normal((12, 2)).astype(np.float32)

    Y_cpu = mpers.compute_persistent_homology(X, max_dim=1, device="cpu")
    Y_gpu = mpers.compute_persistent_homology(X, max_dim=1, device="gpu")

    for i in range(2):
        for j in range(3):
            assert Y_cpu[i, j, 0].is_isomorphic_to(Y_gpu[i, j, 0], atol=_ATOL, rtol=_RTOL)
            assert Y_cpu[i, j, 1].is_isomorphic_to(Y_gpu[i, j, 1], atol=_ATOL, rtol=_RTOL)


def test_ripser_plusplus_device_auto_picks_gpu_when_available():
    X = _rectangle_point_cloud()
    bcs_auto = mpers.compute_persistent_homology(X, max_dim=1, device="auto")
    bcs_gpu = mpers.compute_persistent_homology(X, max_dim=1, device="gpu")
    assert bcs_auto[0].is_isomorphic_to(bcs_gpu[0])
    assert bcs_auto[1].is_isomorphic_to(bcs_gpu[1])


def test_ripser_plusplus_device_invalid_raises():
    X = _rectangle_point_cloud()
    with pytest.raises(ValueError):
        mpers.compute_persistent_homology(X, max_dim=1, device="tpu")
