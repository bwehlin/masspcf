#    Copyright 2024-2026 Bjorn Wehlin
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
import pytest

import masspcf as mpcf
from masspcf.random import Generator
from masspcf.sampling import (
    DistanceWeightedSampler, Gaussian, Uniform, Mixture, SamplingResult,
)


def _make_point_cloud(pts, dtype=mpcf.pcloud64):
    """Wrap a numpy array as a rank-1 PointCloudTensor."""
    X = mpcf.zeros((1,), dtype=dtype)
    X[0] = pts
    return X


class TestWeightFunctions:
    def test_gaussian_peak_at_mean(self):
        g = Gaussian(mean=2.0, sigma=1.0)
        assert g(2.0) == pytest.approx(1.0)
        assert g(2.0) > g(0.0)
        assert g(2.0) > g(5.0)

    def test_gaussian_max(self):
        g = Gaussian(mean=0.0, sigma=1.0)
        assert g.max() == pytest.approx(1.0)

    def test_gaussian_max_in_range_contains_mean(self):
        g = Gaussian(mean=1.0, sigma=1.0)
        assert g.max_in_range(0.0, 2.0) == pytest.approx(1.0)

    def test_gaussian_max_in_range_away_from_mean(self):
        g = Gaussian(mean=0.0, sigma=1.0)
        # Range [3, 5] — peak at x=3
        assert g.max_in_range(3.0, 5.0) == pytest.approx(g(3.0))

    def test_uniform_in_range(self):
        u = Uniform(lo=1.0, hi=3.0)
        assert u(2.0) == pytest.approx(1.0)
        assert u(0.5) == pytest.approx(0.0)
        assert u(3.5) == pytest.approx(0.0)

    def test_uniform_max_in_range(self):
        u = Uniform(lo=1.0, hi=3.0)
        assert u.max_in_range(0.0, 2.0) == pytest.approx(1.0)  # overlaps
        assert u.max_in_range(4.0, 5.0) == pytest.approx(0.0)  # no overlap

    def test_mixture(self):
        g1 = Gaussian(mean=0.0, sigma=1.0)
        g2 = Gaussian(mean=5.0, sigma=0.5)
        mix = Mixture(components=[g1, g2], weights=[0.7, 0.3])
        # At x=0, mostly component 1
        val_at_0 = mix(0.0)
        assert val_at_0 == pytest.approx(0.7 * g1(0.0) + 0.3 * g2(0.0))
        # At x=5, mostly component 2
        val_at_5 = mix(5.0)
        assert val_at_5 == pytest.approx(0.7 * g1(5.0) + 0.3 * g2(5.0))


class TestSamplingWithReplacement:
    def test_basic_gaussian_sampling(self):
        np.random.seed(42)
        pts = np.random.randn(200, 2).astype(np.float64)
        X = _make_point_cloud(pts)
        V = _make_point_cloud(pts)

        dist = Gaussian(mean=0.0, sigma=1.0)
        result = DistanceWeightedSampler(X).sample(V, k=10, dist=dist, replace=True)

        assert result.shape == (200,)
        for i in range(min(5, 200)):
            elem = np.asarray(result[i])
            assert elem.shape == (10, 2)

    def test_sampling_with_radius(self):
        np.random.seed(42)
        pts = np.random.randn(100, 2).astype(np.float64)
        X = _make_point_cloud(pts)

        # Use a single vantage at origin
        vantage_pts = np.zeros((1, 2), dtype=np.float64)
        V = _make_point_cloud(vantage_pts)

        dist = Gaussian(mean=0.0, sigma=0.5)
        result = DistanceWeightedSampler(X).sample(V, k=20, dist=dist, replace=True, radius=1.0)

        sampled = np.asarray(result[0])
        dists = np.linalg.norm(sampled, axis=1)
        # All sampled points should be within radius
        assert np.all(dists <= 1.0 + 1e-6)

    def test_gaussian_favors_nearby_points(self):
        """Points closer to vantage should be sampled more often under tight Gaussian."""
        np.random.seed(123)
        # Place 50 points near origin and 50 far away
        near = np.random.randn(50, 2).astype(np.float64) * 0.1
        far = np.random.randn(50, 2).astype(np.float64) * 0.1 + 10.0
        pts = np.vstack([near, far])
        X = _make_point_cloud(pts)

        vantage_pts = np.zeros((1, 2), dtype=np.float64)
        V = _make_point_cloud(vantage_pts)

        dist = Gaussian(mean=0.0, sigma=0.5)
        gen = Generator(seed=42)
        result = DistanceWeightedSampler(X).sample(V, k=100, dist=dist, replace=True, generator=gen)

        sampled = np.asarray(result[0])
        dists = np.linalg.norm(sampled, axis=1)
        # Most samples should be near origin (distance < 1)
        near_count = np.sum(dists < 1.0)
        assert near_count > 80, f"Expected most samples near origin, got {near_count}/100"

    def test_float32(self):
        np.random.seed(42)
        pts = np.random.randn(50, 3).astype(np.float32)
        X = _make_point_cloud(pts, dtype=mpcf.pcloud32)
        V = _make_point_cloud(pts, dtype=mpcf.pcloud32)

        dist = Gaussian(mean=0.0, sigma=1.0)
        result = DistanceWeightedSampler(X, dtype=mpcf.pcloud32).sample(V, k=5, dist=dist, replace=True)

        assert result.shape == (50,)
        elem = np.asarray(result[0])
        assert elem.shape == (5, 3)
        assert elem.dtype == np.float32


class TestSamplingWithoutReplacement:
    def test_no_duplicates(self):
        np.random.seed(42)
        pts = np.random.randn(50, 2).astype(np.float64)
        X = _make_point_cloud(pts)

        vantage_pts = np.zeros((1, 2), dtype=np.float64)
        V = _make_point_cloud(vantage_pts)

        dist = Gaussian(mean=0.0, sigma=10.0)  # broad so many points are reachable
        gen = Generator(seed=42)
        result = DistanceWeightedSampler(X).sample(V, k=20, dist=dist, replace=False, generator=gen)

        sampled = np.asarray(result[0])
        # Convert to tuples for uniqueness check
        rows = [tuple(row) for row in sampled]
        assert len(set(rows)) == len(rows), "Without-replacement should produce unique points"


class TestUniformDistribution:
    def test_uniform_ring_sampling(self):
        """Uniform(1, 2) should only sample points at distance 1-2."""
        np.random.seed(42)
        # Arrange points at various distances from origin
        angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        radii = np.linspace(0.1, 5.0, 200)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).astype(np.float64)
        X = _make_point_cloud(pts)

        vantage_pts = np.zeros((1, 2), dtype=np.float64)
        V = _make_point_cloud(vantage_pts)

        dist = Uniform(lo=1.0, hi=2.0)
        gen = Generator(seed=42)
        result = DistanceWeightedSampler(X).sample(V, k=30, dist=dist, replace=True, generator=gen)

        sampled = np.asarray(result[0])
        dists = np.linalg.norm(sampled, axis=1)
        assert np.all(dists >= 1.0 - 1e-6), f"Min dist: {dists.min()}"
        assert np.all(dists <= 2.0 + 1e-6), f"Max dist: {dists.max()}"


class TestMixtureDistribution:
    def test_mixture_samples_from_both_modes(self):
        """A mixture with modes at d=0 and d=5 should sample from both regions."""
        np.random.seed(42)
        near = np.random.randn(50, 2).astype(np.float64) * 0.3
        far = np.random.randn(50, 2).astype(np.float64) * 0.3 + 5.0
        pts = np.vstack([near, far])
        X = _make_point_cloud(pts)

        vantage_pts = np.zeros((1, 2), dtype=np.float64)
        V = _make_point_cloud(vantage_pts)

        dist = Mixture(
            components=[Gaussian(mean=0.0, sigma=2.0), Gaussian(mean=5.0, sigma=2.0)],
            weights=[0.5, 0.5],
        )
        gen = Generator(seed=42)
        result = DistanceWeightedSampler(X).sample(V, k=500, dist=dist, replace=True, generator=gen)

        sampled = np.asarray(result[0])
        dists = np.linalg.norm(sampled, axis=1)
        near_count = np.sum(dists < 2.0)
        far_count = np.sum(dists > 3.0)
        # Both modes should be represented
        assert near_count >= 10, f"Expected samples near origin, got {near_count}"
        assert far_count >= 10, f"Expected samples far from origin, got {far_count}"


    def test_heterogeneous_mixture(self):
        """A Mixture of Gaussian + Uniform should sample from both component types."""
        # Points at various distances from origin
        pts = np.array([
            [0.1, 0.0], [0.2, 0.0],   # near (distance ~0.1-0.2)
            [1.5, 0.0], [1.8, 0.0],   # mid-ring (distance 1.5-1.8)
            [5.0, 0.0], [6.0, 0.0],   # far (distance 5-6)
        ], dtype=np.float64)
        X = _make_point_cloud(pts)
        vantage_pts = np.array([[0.0, 0.0]], dtype=np.float64)
        V = _make_point_cloud(vantage_pts)

        # Gaussian near origin + Uniform ring at distance 1-2
        dist = Mixture(
            components=[Gaussian(mean=0.0, sigma=0.3), Uniform(lo=1.0, hi=2.0)],
            weights=[0.5, 0.5],
        )
        gen = Generator(seed=42)
        result = DistanceWeightedSampler(X).sample(V, k=500, dist=dist, replace=True, generator=gen)

        sampled = np.asarray(result[0])
        dists = np.linalg.norm(sampled, axis=1)
        near_count = np.sum(dists < 0.5)
        ring_count = np.sum((dists >= 1.0) & (dists <= 2.0))
        far_count = np.sum(dists > 4.0)
        assert near_count > 10, f"Expected samples from Gaussian mode, got {near_count}"
        assert ring_count > 10, f"Expected samples from Uniform ring, got {ring_count}"
        assert far_count == 0, f"Expected no far samples, got {far_count}"


class TestAdaptiveSampling:
    def test_tight_mixture_adapts(self):
        """Adaptive escalation should handle tight separated modes automatically."""
        np.random.seed(42)
        near = np.random.randn(50, 2).astype(np.float64) * 0.3
        far = np.random.randn(50, 2).astype(np.float64) * 0.3 + [5.0, 0.0]
        pts = np.vstack([near, far])
        X = _make_point_cloud(pts)

        vantage_pts = np.zeros((1, 2), dtype=np.float64)
        V = _make_point_cloud(vantage_pts)

        dist = Mixture(
            components=[Gaussian(mean=0.0, sigma=0.1), Gaussian(mean=5.0, sigma=0.1)],
            weights=[0.5, 0.5],
        )
        gen = Generator(seed=42)
        # Use aggressive escalation to ensure both modes are reached
        result = DistanceWeightedSampler(X).sample(V, k=500, dist=dist, replace=True,
                                                    generator=gen, escalation_threshold=20)

        sampled = np.asarray(result[0])
        dists = np.linalg.norm(sampled, axis=1)
        near_count = np.sum(dists < 2.0)
        far_count = np.sum(dists > 3.0)
        assert near_count >= 10, f"Expected samples near origin, got {near_count}"
        assert far_count >= 10, f"Expected samples far from origin, got {far_count}"

        # Adaptive escalation should have been triggered for this hard distribution
        assert not result.diagnostics.all_exact

    def test_custom_stages(self):
        """Custom escalation stages should be accepted."""
        np.random.seed(42)
        pts = np.random.randn(50, 2).astype(np.float64)
        X = _make_point_cloud(pts)
        V = _make_point_cloud(np.zeros((1, 2), dtype=np.float64))

        dist = Gaussian(mean=0.0, sigma=1.0)
        gen = Generator(seed=42)
        result = DistanceWeightedSampler(X).sample(V, k=10, dist=dist, replace=True,
                                                    generator=gen, stages=(0.0, 1.0),
                                                    escalation_threshold=50, max_attempts=500)
        assert result.shape == (1,)


class TestDiagnostics:
    def test_easy_distribution_is_exact(self):
        """A broad Gaussian on a small cloud should not trigger escalation."""
        np.random.seed(42)
        pts = np.random.randn(50, 2).astype(np.float64)
        X = _make_point_cloud(pts)
        V = _make_point_cloud(np.zeros((1, 2), dtype=np.float64))

        dist = Gaussian(mean=0.0, sigma=5.0)
        gen = Generator(seed=42)
        result = DistanceWeightedSampler(X).sample(V, k=20, dist=dist, replace=True, generator=gen)

        diag = result.diagnostics
        assert diag.all_exact
        assert diag.biased_vantage_count == 0
        assert len(diag.acceptance_rate) == 1
        assert diag.acceptance_rate[0] > 0
        assert len(diag.total_attempts) == 1
        assert diag.total_attempts[0] >= 20
        assert len(diag.biased) == 1
        assert not diag.biased[0]

    def test_result_is_sampling_result(self):
        """DistanceWeightedSampler.sample() should return a SamplingResult."""
        np.random.seed(42)
        pts = np.random.randn(50, 2).astype(np.float64)
        X = _make_point_cloud(pts)
        V = _make_point_cloud(np.zeros((1, 2), dtype=np.float64))

        dist = Gaussian(mean=0.0, sigma=1.0)
        result = DistanceWeightedSampler(X).sample(V, k=10, dist=dist, replace=True)
        assert isinstance(result, SamplingResult)

    def test_backward_compat_delegates(self):
        """SamplingResult should delegate IndexedPointCloudTensor interface."""
        np.random.seed(42)
        pts = np.random.randn(30, 2).astype(np.float64)
        X = _make_point_cloud(pts)
        V = _make_point_cloud(np.zeros((2, 2), dtype=np.float64))

        dist = Gaussian(mean=0.0, sigma=2.0)
        gen = Generator(seed=42)
        result = DistanceWeightedSampler(X).sample(V, k=5, dist=dist, replace=True, generator=gen)

        # Shape / len / ndim
        assert result.shape == (2,)
        assert len(result) == 2
        assert result.ndim == 1

        # Indexing
        elem = np.asarray(result[0])
        assert elem.shape == (5, 2)

        # Iteration
        count = sum(1 for _ in result)
        assert count == 2

        # Properties
        assert result.indices is not None
        assert result.source is not None


class TestEdgeCases:
    def test_uniform_hi_negative_returns_empty(self):
        """Uniform with hi < 0 has zero weight everywhere; samples should be empty."""
        pts = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        X = _make_point_cloud(pts)
        vantage_pts = np.array([[0.0, 0.0]], dtype=np.float64)
        V = _make_point_cloud(vantage_pts)

        dist = Uniform(lo=-3.0, hi=-1.0)
        gen = Generator(seed=42)
        result = DistanceWeightedSampler(X).sample(V, k=10, dist=dist, replace=True, generator=gen)

        sampled = np.asarray(result[0])
        assert sampled.shape[0] == 0, f"Expected empty, got shape {sampled.shape}"

    def test_uniform_lo_gt_hi_raises(self):
        """Uniform with lo > hi should raise ValueError."""
        with pytest.raises(ValueError, match="lo.*must be <= hi"):
            Uniform(lo=3.0, hi=1.0)

    def test_radius_excludes_all_returns_empty(self):
        """Ball radius smaller than nearest point distance returns empty."""
        pts = np.array([[10.0, 0.0], [20.0, 0.0]], dtype=np.float64)
        X = _make_point_cloud(pts)
        vantage_pts = np.array([[0.0, 0.0]], dtype=np.float64)
        V = _make_point_cloud(vantage_pts)

        dist = Gaussian(mean=0.0, sigma=1.0)
        gen = Generator(seed=42)
        result = DistanceWeightedSampler(X).sample(V, k=10, dist=dist, replace=True, generator=gen, radius=0.1)

        sampled = np.asarray(result[0])
        assert sampled.shape[0] == 0, f"Expected empty, got shape {sampled.shape}"


class TestUnbiasedSampling:
    def test_chi_squared_gaussian(self):
        """Chi-squared test: samples should match the exact categorical distribution."""
        from scipy import stats

        # Small point cloud with known positions
        np.random.seed(123)
        pts = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
            [3.0, 3.0],
            [4.0, 0.0],
            [0.0, 4.0],
            [2.5, 1.0],
            [1.0, 2.5],
            [3.5, 1.5],
            [1.5, 3.5],
            [2.5, 2.5],
        ], dtype=np.float64)
        n_points = len(pts)

        X = _make_point_cloud(pts)

        # Vantage at origin
        vantage_pts = np.array([[0.0, 0.0]], dtype=np.float64)
        V = _make_point_cloud(vantage_pts)

        sigma = 2.0
        dist = Gaussian(mean=0.0, sigma=sigma)

        # Compute exact weights
        distances = np.linalg.norm(pts - vantage_pts[0], axis=1)
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        expected_probs = weights / weights.sum()

        # Draw many samples
        n_samples = 100_000
        gen = Generator(seed=7)
        result = DistanceWeightedSampler(X).sample(V, k=n_samples, dist=dist, replace=True, generator=gen)
        sampled = np.asarray(result[0])

        # Count how often each source point was sampled
        observed = np.zeros(n_points, dtype=int)
        for s in range(n_samples):
            pt = sampled[s]
            # Find matching source point
            diffs = np.linalg.norm(pts - pt, axis=1)
            idx = np.argmin(diffs)
            assert diffs[idx] < 1e-10, f"Sample {s} doesn't match any source point"
            observed[idx] += 1

        # Verify sampling was exact (no adaptive escalation)
        assert result.diagnostics.all_exact, "Unbiased test should not trigger escalation"

        # Chi-squared test at 1% significance
        expected_counts = expected_probs * n_samples
        chi2, p_value = stats.chisquare(observed, f_exp=expected_counts)
        assert p_value > 0.01, (
            f"Chi-squared test failed: p={p_value:.6f}, chi2={chi2:.1f}. "
            f"Observed: {observed}, Expected: {np.round(expected_counts).astype(int)}"
        )


class TestNaiveSampling:
    def test_naive_basic_gaussian(self):
        np.random.seed(42)
        pts = np.random.randn(200, 2).astype(np.float64)
        X = _make_point_cloud(pts)
        V = _make_point_cloud(pts)

        dist = Gaussian(mean=0.0, sigma=1.0)
        result = DistanceWeightedSampler(X).sample(V, k=10, dist=dist, replace=True, algorithm="naive")

        assert result.shape == (200,)
        for i in range(min(5, 200)):
            elem = np.asarray(result[i])
            assert elem.shape == (10, 2)

    def test_naive_without_replacement(self):
        np.random.seed(42)
        pts = np.random.randn(50, 2).astype(np.float64)
        X = _make_point_cloud(pts)
        V = _make_point_cloud(np.zeros((1, 2), dtype=np.float64))

        dist = Gaussian(mean=0.0, sigma=10.0)
        gen = Generator(seed=42)
        result = DistanceWeightedSampler(X).sample(V, k=20, dist=dist, replace=False,
                                                    generator=gen, algorithm="naive")

        sampled = np.asarray(result[0])
        rows = [tuple(row) for row in sampled]
        assert len(set(rows)) == len(rows), "Without-replacement should produce unique points"

    def test_naive_with_radius(self):
        np.random.seed(42)
        pts = np.random.randn(100, 2).astype(np.float64)
        X = _make_point_cloud(pts)
        V = _make_point_cloud(np.zeros((1, 2), dtype=np.float64))

        dist = Gaussian(mean=0.0, sigma=0.5)
        result = DistanceWeightedSampler(X).sample(V, k=20, dist=dist, replace=True,
                                                    radius=1.0, algorithm="naive")

        sampled = np.asarray(result[0])
        dists = np.linalg.norm(sampled, axis=1)
        assert np.all(dists <= 1.0 + 1e-6)

    def test_naive_diagnostics(self):
        np.random.seed(42)
        pts = np.random.randn(50, 2).astype(np.float64)
        X = _make_point_cloud(pts)
        V = _make_point_cloud(np.zeros((1, 2), dtype=np.float64))

        dist = Gaussian(mean=0.0, sigma=5.0)
        gen = Generator(seed=42)
        result = DistanceWeightedSampler(X).sample(V, k=20, dist=dist, replace=True,
                                                    generator=gen, algorithm="naive")

        diag = result.diagnostics
        assert diag.acceptance_rate[0] == pytest.approx(1.0)
        assert not diag.biased[0]
        assert diag.all_exact

    def test_naive_chi_squared(self):
        """Chi-squared test: naive samples should match exact categorical distribution."""
        from scipy import stats

        pts = np.array([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
            [0.0, 1.0], [0.0, 2.0], [1.0, 1.0], [2.0, 2.0],
            [0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5],
            [3.0, 3.0], [4.0, 0.0], [0.0, 4.0], [2.5, 1.0],
            [1.0, 2.5], [3.5, 1.5], [1.5, 3.5], [2.5, 2.5],
        ], dtype=np.float64)
        n_points = len(pts)

        X = _make_point_cloud(pts)
        vantage_pts = np.array([[0.0, 0.0]], dtype=np.float64)
        V = _make_point_cloud(vantage_pts)

        sigma = 2.0
        dist = Gaussian(mean=0.0, sigma=sigma)

        distances = np.linalg.norm(pts - vantage_pts[0], axis=1)
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        expected_probs = weights / weights.sum()

        n_samples = 100_000
        gen = Generator(seed=7)
        result = DistanceWeightedSampler(X).sample(V, k=n_samples, dist=dist, replace=True,
                                                    generator=gen, algorithm="naive")
        sampled = np.asarray(result[0])

        observed = np.zeros(n_points, dtype=int)
        for s in range(n_samples):
            pt = sampled[s]
            diffs = np.linalg.norm(pts - pt, axis=1)
            idx = np.argmin(diffs)
            assert diffs[idx] < 1e-10, f"Sample {s} doesn't match any source point"
            observed[idx] += 1

        assert result.diagnostics.all_exact

        expected_counts = expected_probs * n_samples
        chi2, p_value = stats.chisquare(observed, f_exp=expected_counts)
        assert p_value > 0.01, (
            f"Chi-squared test failed: p={p_value:.6f}, chi2={chi2:.1f}. "
            f"Observed: {observed}, Expected: {np.round(expected_counts).astype(int)}"
        )


class TestDistanceWeightedSampler:
    def test_reuse_across_distributions(self):
        """Build sampler once, sample with different distributions."""
        np.random.seed(42)
        pts = np.random.randn(100, 2).astype(np.float64)
        X = _make_point_cloud(pts)

        vantage_pts = np.zeros((1, 2), dtype=np.float64)
        V = _make_point_cloud(vantage_pts)

        sampler = DistanceWeightedSampler(X)

        r1 = sampler.sample(V, k=10, dist=Gaussian(mean=0.0, sigma=1.0))
        r2 = sampler.sample(V, k=10, dist=Uniform(lo=0.5, hi=2.0))

        assert r1.shape == (1,)
        assert r2.shape == (1,)
        assert np.asarray(r1[0]).shape == (10, 2)
        assert np.asarray(r2[0]).shape == (10, 2)

    def test_reuse_across_vantage_sets(self):
        """Build sampler once, sample with different vantage sets."""
        np.random.seed(42)
        pts = np.random.randn(100, 3).astype(np.float64)
        X = _make_point_cloud(pts)

        sampler = DistanceWeightedSampler(X)
        dist = Gaussian(mean=0.0, sigma=2.0)

        V1 = _make_point_cloud(np.zeros((5, 3), dtype=np.float64))
        V2 = _make_point_cloud(np.ones((3, 3), dtype=np.float64))

        r1 = sampler.sample(V1, k=10, dist=dist)
        r2 = sampler.sample(V2, k=10, dist=dist)

        assert r1.shape == (5,)
        assert r2.shape == (3,)

    def test_properties(self):
        np.random.seed(42)
        pts = np.random.randn(50, 4).astype(np.float64)
        X = _make_point_cloud(pts)

        sampler = DistanceWeightedSampler(X)
        assert sampler.dim == 4
        assert sampler.n_points == 50
