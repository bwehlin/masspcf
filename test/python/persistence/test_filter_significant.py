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

import masspcf as mpcf
import masspcf.persistence as mpers


def test_empty_barcode():
    bc = mpers.Barcode(np.zeros((0, 2)))
    result = mpers.filter_significant_bars(bc)
    assert len(result) == 0


def test_single_bar_birth_zero_always_kept():
    bc = mpers.Barcode(np.array([[0.0, 5.0]]))
    result = mpers.filter_significant_bars(bc)
    assert len(result) == 1
    arr = np.asarray(result)
    assert arr[0, 0] == 0.0
    assert arr[0, 1] == 5.0


def test_infinite_death_bar_always_kept():
    bc = mpers.Barcode(np.array([
        [1.0, np.inf],
        [1.0, 1.01],
        [2.0, 2.01],
    ]))
    result = mpers.filter_significant_bars(bc)
    arr = np.asarray(result)
    has_inf = np.any(np.isinf(arr[:, 1]))
    assert has_inf


def test_clear_signal_among_noise():
    bars = [[1.0, 100.0]]  # signal: pi = 100
    for i in range(50):
        birth = 1.0 + i * 0.1
        bars.append([birth, birth * 1.01])  # noise: pi = 1.01
    bc = mpers.Barcode(np.array(bars))
    result = mpers.filter_significant_bars(bc)

    arr = np.asarray(result)
    assert len(result) >= 1

    # The signal bar should be present
    signal_found = np.any((arr[:, 0] == 1.0) & (arr[:, 1] == 100.0))
    assert signal_found

    # Most noise should be removed
    assert len(result) < 10


def test_all_noise_filters_all():
    bars = []
    for i in range(30):
        birth = 1.0 + i * 0.1
        bars.append([birth, birth * 1.05])
    bc = mpers.Barcode(np.array(bars))
    result = mpers.filter_significant_bars(bc)
    assert len(result) == 0


def test_tensor_matches_elementwise():
    bcs = mpcf.zeros((3,), dtype=mpcf.barcode64)

    # 0: clear signal
    bars0 = [[1.0, 100.0]]
    for i in range(20):
        birth = 1.0 + i * 0.1
        bars0.append([birth, birth * 1.01])
    bcs[0] = mpers.Barcode(np.array(bars0))

    # 1: all noise
    bars1 = []
    for i in range(15):
        birth = 1.0 + i * 0.1
        bars1.append([birth, birth * 1.03])
    bcs[1] = mpers.Barcode(np.array(bars1))

    # 2: empty
    bcs[2] = mpers.Barcode(np.zeros((0, 2)))

    result = mpers.filter_significant_bars(bcs)
    assert result.shape == bcs.shape

    for i in range(3):
        expected = mpers.filter_significant_bars(bcs[i])
        assert result[i].is_isomorphic_to(expected), f"Mismatch at index {i}"


def test_circle_point_cloud_has_one_significant_h1():
    """Integration test: a circle should have exactly 1 significant H1 cycle.

    Uses a noisier circle with more points to produce enough H1 bars
    for the statistical test to have power.
    """
    rng = np.random.default_rng(42)
    n = 200
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    noise = 0.15 * rng.standard_normal((n, 2))
    pts = np.column_stack([np.cos(theta), np.sin(theta)]) + noise

    barcodes = mpers.compute_persistent_homology(pts, max_dim=1)

    h1 = barcodes[1]
    assert len(h1) > 5, f"Need enough bars for the test to have power, got {len(h1)}"

    h1_filtered = mpers.filter_significant_bars(h1)

    assert len(h1_filtered) == 1
