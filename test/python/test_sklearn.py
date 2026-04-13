"""Tests for masspcf.sklearn transformers."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import masspcf as mpcf
from masspcf.sklearn import (
    Mean,
    PcfKernelTransformer,
    PersistentHomology,
    StableRank,
    TimeDelayEmbedding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_arrays():
    """Small synthetic dataset: 8 instances, 20 time steps, 2 channels."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((8, 20, 2))
    y = np.array(["a", "a", "b", "b", "a", "a", "b", "b"])
    return X, y


@pytest.fixture()
def sample_tensor(sample_arrays):
    X, _ = sample_arrays
    return mpcf.TimeSeriesTensor([
        mpcf.TimeSeries(X[i], start_time=0.0, time_step=1.0)
        for i in range(len(X))
    ])


# ---------------------------------------------------------------------------
# TimeDelayEmbedding
# ---------------------------------------------------------------------------


class TestTimeDelayEmbedding:
    def test_from_numpy(self, sample_arrays):
        X, _ = sample_arrays
        emb = TimeDelayEmbedding(dimension=2, delay=2.0, time_step=1.0)
        clouds = emb.fit_transform(X)
        assert clouds.shape[0] == 8

    def test_from_tensor(self, sample_tensor):
        emb = TimeDelayEmbedding(dimension=2, delay=2.0)
        clouds = emb.fit_transform(sample_tensor)
        assert clouds.shape[0] == 8

    def test_with_windowing(self, sample_arrays):
        X, _ = sample_arrays
        emb = TimeDelayEmbedding(dimension=2, delay=2.0, time_step=1.0,
                                 window=8.0, stride=4.0)
        clouds = emb.fit_transform(X)
        assert len(clouds.shape) == 2
        assert clouds.shape[0] == 8
        assert clouds.shape[1] > 1

    def test_get_params(self):
        emb = TimeDelayEmbedding(dimension=3, delay=0.5, window=2.0)
        params = emb.get_params()
        assert params["dimension"] == 3
        assert params["delay"] == 0.5
        assert params["window"] == 2.0

    def test_set_params(self):
        emb = TimeDelayEmbedding(dimension=2, delay=1.0)
        emb.set_params(delay=0.5)
        assert emb.delay == 0.5


# ---------------------------------------------------------------------------
# PersistentHomology
# ---------------------------------------------------------------------------


class TestPersistentHomology:
    def test_transform(self, sample_arrays):
        X, _ = sample_arrays
        emb = TimeDelayEmbedding(dimension=2, delay=2.0, time_step=1.0)
        clouds = emb.transform(X)

        ph = PersistentHomology(max_dim=1)
        barcodes = ph.fit_transform(clouds)
        # Last axis should be max_dim + 1
        assert barcodes.shape[-1] == 2

    def test_max_dim_0(self, sample_arrays):
        X, _ = sample_arrays
        clouds = TimeDelayEmbedding(
            dimension=2, delay=2.0, time_step=1.0).transform(X)

        ph = PersistentHomology(max_dim=0)
        barcodes = ph.transform(clouds)
        assert barcodes.shape[-1] == 1


# ---------------------------------------------------------------------------
# StableRank
# ---------------------------------------------------------------------------


class TestStableRank:
    def test_no_selection(self, sample_arrays):
        X, _ = sample_arrays
        clouds = TimeDelayEmbedding(
            dimension=2, delay=2.0, time_step=1.0).transform(X)
        barcodes = PersistentHomology(max_dim=1).transform(clouds)

        sr = StableRank()
        sranks = sr.fit_transform(barcodes)
        assert sranks.shape == barcodes.shape

    def test_dim_selection(self, sample_arrays):
        X, _ = sample_arrays
        clouds = TimeDelayEmbedding(
            dimension=2, delay=2.0, time_step=1.0).transform(X)
        barcodes = PersistentHomology(max_dim=1).transform(clouds)

        sr = StableRank(dim=1)
        sranks = sr.transform(barcodes)
        # Should have dropped the last axis
        assert len(sranks.shape) == len(barcodes.shape) - 1

    def test_dim_with_mean_reduction(self, sample_arrays):
        X, _ = sample_arrays
        clouds = TimeDelayEmbedding(
            dimension=2, delay=2.0, time_step=1.0,
            window=8.0, stride=4.0).transform(X)
        barcodes = PersistentHomology(max_dim=1).transform(clouds)

        # Shape: (8, n_windows, 2)
        sr = StableRank(dim=1)
        sranks = sr.transform(barcodes)
        # Shape: (8, n_windows)
        reduced = Mean().transform(sranks)
        # Should be (8,) -- one PCF per instance
        assert reduced.shape == (8,)

    def test_mean_explicit_dim(self, sample_arrays):
        X, _ = sample_arrays
        clouds = TimeDelayEmbedding(
            dimension=2, delay=2.0, time_step=1.0,
            window=8.0, stride=4.0).transform(X)
        barcodes = PersistentHomology(max_dim=1).transform(clouds)

        sranks = StableRank(dim=1).transform(barcodes)
        reduced = Mean(dim=1).transform(sranks)
        assert reduced.shape == (8,)


# ---------------------------------------------------------------------------
# PcfKernelTransformer
# ---------------------------------------------------------------------------


class TestPcfKernelTransformer:
    def test_fit_transform_symmetric(self):
        X = mpcf.random.noisy_sin(shape=(5,))
        kt = PcfKernelTransformer()
        K = kt.fit_transform(X)
        assert isinstance(K, np.ndarray)
        assert K.shape == (5, 5)
        np.testing.assert_allclose(K, K.T)

    def test_transform_cross_kernel(self):
        X = mpcf.random.noisy_sin(shape=(5,))
        Y = mpcf.random.noisy_sin(shape=(3,))
        kt = PcfKernelTransformer()
        kt.fit(X)
        K_cross = kt.transform(Y)
        assert isinstance(K_cross, np.ndarray)
        assert K_cross.shape == (3, 5)

    def test_transform_before_fit_raises(self):
        from sklearn.exceptions import NotFittedError
        kt = PcfKernelTransformer()
        X = mpcf.random.noisy_sin(shape=(3,))
        with pytest.raises(NotFittedError):
            kt.transform(X)

    def test_cross_matches_full(self):
        X = mpcf.random.noisy_sin(shape=(4,))
        Y = mpcf.random.noisy_sin(shape=(3,))

        kt = PcfKernelTransformer()
        kt.fit(X)
        K_cross = kt.transform(Y)

        # Compare against concatenated full kernel
        concat = mpcf.PcfTensor(
            [X[i] for i in range(4)] + [Y[i] for i in range(3)])
        K_full = mpcf.l2_kernel(concat).to_dense()
        np.testing.assert_allclose(K_cross, K_full[4:, :4], rtol=1e-5)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_fit_predict(self, sample_arrays):
        X, y = sample_arrays
        pipe = Pipeline([
            ("embed", TimeDelayEmbedding(
                dimension=2, delay=2.0, time_step=1.0,
                window=8.0, stride=4.0)),
            ("ph", PersistentHomology(max_dim=1)),
            ("sr", StableRank(dim=1)),
            ("mean", Mean()),
            ("kernel", PcfKernelTransformer()),
            ("svc", SVC(kernel="precomputed")),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset(set(y))

    def test_separate_train_test(self, sample_arrays):
        X, y = sample_arrays
        X_train, X_test = X[:6], X[6:]
        y_train, y_test = y[:6], y[6:]

        pipe = Pipeline([
            ("embed", TimeDelayEmbedding(
                dimension=2, delay=2.0, time_step=1.0,
                window=8.0, stride=4.0)),
            ("ph", PersistentHomology(max_dim=1)),
            ("sr", StableRank(dim=1)),
            ("mean", Mean()),
            ("kernel", PcfKernelTransformer()),
            ("svc", SVC(kernel="precomputed")),
        ])
        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)
        assert 0.0 <= score <= 1.0

    def test_get_params_nested(self):
        pipe = Pipeline([
            ("embed", TimeDelayEmbedding(dimension=2, delay=0.3)),
            ("ph", PersistentHomology(max_dim=1)),
            ("sr", StableRank(dim=1)),
            ("mean", Mean()),
            ("kernel", PcfKernelTransformer()),
            ("svc", SVC(kernel="precomputed")),
        ])
        params = pipe.get_params()
        assert params["embed__dimension"] == 2
        assert params["embed__delay"] == 0.3
        assert params["ph__max_dim"] == 1
        assert params["sr__dim"] == 1


# ---------------------------------------------------------------------------
# Coverage gap tests
# ---------------------------------------------------------------------------


class TestMeanStandalone:
    def test_default_dim_reduces_last(self, sample_arrays):
        """Mean with default dim reduces the last axis."""
        X, _ = sample_arrays
        clouds = TimeDelayEmbedding(
            dimension=2, delay=2.0, time_step=1.0,
            window=8.0, stride=4.0).transform(X)
        barcodes = PersistentHomology(max_dim=1).transform(clouds)
        sranks = StableRank().transform(barcodes)
        # sranks has shape (8, n_windows, 2) -- mean over last axis
        n_dims = len(sranks.shape)
        reduced = Mean().transform(sranks)
        assert len(reduced.shape) == n_dims - 1
        assert reduced.shape[0] == sranks.shape[0]

    def test_dim_0_reduces_first(self, sample_arrays):
        """Mean with dim=0 reduces the first axis."""
        X, _ = sample_arrays
        clouds = TimeDelayEmbedding(
            dimension=2, delay=2.0, time_step=1.0).transform(X)
        barcodes = PersistentHomology(max_dim=1).transform(clouds)
        sranks = StableRank().transform(barcodes)
        n_dims = len(sranks.shape)
        reduced = Mean(dim=0).transform(sranks)
        assert len(reduced.shape) == n_dims - 1
        assert reduced.shape[0] == sranks.shape[1]


class TestTimeDelayEmbeddingMultiChannel:
    def test_from_numpy_multichannel(self, sample_arrays):
        """Numpy input with shape (n_instances, n_times, n_channels)
        converts each instance to a multi-channel TimeSeries."""
        X, _ = sample_arrays  # shape (8, 20, 2)
        emb = TimeDelayEmbedding(dimension=2, delay=2.0, time_step=1.0)
        clouds = emb.transform(X)
        assert clouds.shape[0] == 8
        # Each cloud should have dimension * n_channels = 4 columns
        cloud = np.asarray(clouds[0])
        assert cloud.shape[1] == 2 * 2  # dimension * n_channels


class TestPcfKernelFitTransformConsistency:
    def test_fit_transform_matches_fit_then_transform(self):
        """fit_transform (optimized symmetric path) should produce
        the same values as fit + transform."""
        X = mpcf.random.noisy_sin(shape=(5,))
        kt1 = PcfKernelTransformer()
        K_opt = kt1.fit_transform(X)

        kt2 = PcfKernelTransformer()
        kt2.fit(X)
        K_seq = kt2.transform(X)

        np.testing.assert_allclose(K_opt, K_seq, rtol=1e-5)
