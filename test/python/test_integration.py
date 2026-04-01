"""End-to-end integration tests exercising full TDA pipelines."""

import numpy as np
import numpy.testing as npt

import masspcf as mpcf
import masspcf.persistence as mpers


def test_pcf_generation_to_distance_to_persistence():
    """Full pipeline: generate random PCFs -> pdist -> persistent homology -> barcode summaries."""
    np.random.seed(42)

    # Step 1: Generate random PCFs
    X = mpcf.random.noisy_sin((8,), dtype=mpcf.pcf64)
    assert X.shape == (8,)

    # Step 2: Compute pairwise L1 distance matrix
    dm = mpcf.pdist(X, p=1, verbose=False)
    assert dm.size == 8
    # Distance matrix should be symmetric with zero diagonal
    dense = dm.to_dense()
    npt.assert_allclose(dense, dense.T, atol=1e-10)
    npt.assert_allclose(np.diag(dense), 0.0, atol=1e-10)

    # Step 3: Compute persistent homology from distance matrix
    bcs = mpers.compute_persistent_homology(dm, maxDim=1, verbose=False)
    assert isinstance(bcs, mpers.BarcodeTensor)
    assert bcs.shape == (2,)  # H0 and H1

    # Step 4: Convert barcodes to summary functions
    sr_h0 = mpers.barcode_to_stable_rank(bcs[0])
    assert isinstance(sr_h0, mpcf.Pcf)

    bc_h1 = mpers.barcode_to_betti_curve(bcs[1])
    assert isinstance(bc_h1, mpcf.Pcf)

    ap_h0 = mpers.barcode_to_accumulated_persistence(bcs[0])
    assert isinstance(ap_h0, mpcf.Pcf)


def test_point_cloud_to_persistence_to_stable_rank():
    """Pipeline: point clouds -> persistence -> stable rank as PCFs -> pdist on stable ranks."""
    np.random.seed(7)

    # Step 1: Generate point clouds
    n_clouds = 5
    pclouds = mpcf.zeros((n_clouds,), dtype=mpcf.pcloud64)
    for i in range(n_clouds):
        pclouds[i] = np.random.randn(15, 2)

    # Step 2: Compute persistence
    bcs = mpers.compute_persistent_homology(pclouds, maxDim=1, verbose=False)
    assert bcs.shape == (n_clouds, 2)

    # Step 3: Extract H1 stable ranks as PCFs
    srs = mpcf.zeros((n_clouds,), dtype=mpcf.pcf64)
    for i in range(n_clouds):
        srs[i] = mpers.barcode_to_stable_rank(bcs[i, 1])

    # Step 4: Compute distance matrix between stable ranks
    dm = mpcf.pdist(srs, p=1, verbose=False)
    assert dm.size == n_clouds
    dense = dm.to_dense()
    # Basic metric properties
    assert np.all(dense >= -1e-10)
    npt.assert_allclose(np.diag(dense), 0.0, atol=1e-10)
    npt.assert_allclose(dense, dense.T, atol=1e-10)


def test_cdist_between_two_groups():
    """Pipeline: two groups of PCFs -> cdist -> verify shape and non-negativity."""
    np.random.seed(99)

    X = mpcf.random.noisy_sin((4,), dtype=mpcf.pcf64)
    Y = mpcf.random.noisy_cos((3,), dtype=mpcf.pcf64)

    D = mpcf.cdist(X, Y, p=2, verbose=False)
    assert D.shape == (4, 3)
    dense = np.asarray(D)
    assert np.all(dense >= -1e-10)


def test_pcf_arithmetic_then_distance():
    """Pipeline: create PCFs with arithmetic -> compute distances."""
    f = mpcf.Pcf(np.array([[0.0, 1.0], [1.0, 0.0]]))
    g = mpcf.Pcf(np.array([[0.0, 2.0], [1.0, 0.0]]))

    # Arithmetic
    h = f + g  # h = 3 on [0,1)
    k = g - f  # k = 1 on [0,1)

    # Distance between h and k
    d = mpcf.lp_distance(h, k, p=1)
    # |3 - 1| * 1 = 2
    assert d == npt.assert_allclose(d, 2.0, atol=1e-10) or d == 2.0


def test_io_roundtrip_preserves_distances(tmp_path):
    """Pipeline: save/load PCF tensor -> distances should be preserved."""
    X = mpcf.random.noisy_sin((6,), dtype=mpcf.pcf64)
    dm_before = mpcf.pdist(X, verbose=False)

    path = str(tmp_path / "pcfs.mpcf")
    mpcf.save(X, path)
    X_loaded = mpcf.load(path)

    dm_after = mpcf.pdist(X_loaded, verbose=False)

    npt.assert_allclose(dm_before.to_dense(), dm_after.to_dense(), atol=1e-12)
