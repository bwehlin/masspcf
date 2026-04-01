"""Robustness tests for serialization: corrupted data, truncation, edge cases."""

import io

import numpy as np
import pytest

import masspcf as mpcf
import masspcf.persistence as mpers


# --- Corrupted data ---


def test_load_from_empty_bytes_raises():
    """Loading from empty bytes should raise, not crash."""
    buf = io.BytesIO(b"")
    with pytest.raises((RuntimeError, ValueError, EOFError)):
        mpcf.load(buf)


def test_load_from_random_bytes_raises():
    """Loading from random garbage should raise, not crash."""
    buf = io.BytesIO(b"\x00\x01\x02\x03\xff\xfe\xfd" * 10)
    with pytest.raises((RuntimeError, ValueError, EOFError)):
        mpcf.load(buf)


def test_load_truncated_data_raises():
    """Saving then truncating should raise on load."""
    X = mpcf.random.noisy_sin((3,), dtype=mpcf.pcf64)
    buf = io.BytesIO()
    mpcf.save(X, buf)
    data = buf.getvalue()

    # Truncate to half
    truncated = io.BytesIO(data[: len(data) // 2])
    with pytest.raises((RuntimeError, ValueError, EOFError)):
        mpcf.load(truncated)


# --- Edge-case tensors ---


def test_save_load_single_element_tensor():
    """Roundtrip a (1,) tensor."""
    X = mpcf.zeros((1,), dtype=mpcf.pcf64)
    X[0] = mpcf.Pcf(np.array([[0.0, 42.0], [1.0, 0.0]]))

    buf = io.BytesIO()
    mpcf.save(X, buf)
    buf.seek(0)
    Y = mpcf.load(buf)

    assert Y.shape == (1,)
    assert Y[0] == X[0]


def test_save_load_high_dimensional_tensor():
    """Roundtrip a 4D tensor."""
    X = mpcf.zeros((2, 2, 2, 2), dtype=mpcf.pcf32)

    buf = io.BytesIO()
    mpcf.save(X, buf)
    buf.seek(0)
    Y = mpcf.load(buf)

    assert Y.shape == (2, 2, 2, 2)
    assert Y.dtype == mpcf.pcf32


def test_save_load_pcf_object():
    """Roundtrip a single Pcf object."""
    f = mpcf.Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.0]]))

    buf = io.BytesIO()
    mpcf.save(f, buf)
    buf.seek(0)
    g = mpcf.load(buf)

    assert isinstance(g, mpcf.Pcf)
    assert f == g


def test_save_load_barcode_object():
    """Roundtrip a single Barcode object."""
    bc = mpers.Barcode(np.array([[0.0, 1.0], [0.5, 3.0]]))

    buf = io.BytesIO()
    mpcf.save(bc, buf)
    buf.seek(0)
    bc2 = mpcf.load(buf)

    assert isinstance(bc2, mpers.Barcode)
    assert bc.is_isomorphic_to(bc2)


def test_save_load_distance_matrix():
    """Roundtrip a DistanceMatrix object."""
    dm = mpcf.DistanceMatrix(4, dtype=mpcf.float64)
    dm[1, 0] = 1.0
    dm[2, 0] = 2.0
    dm[2, 1] = 3.0
    dm[3, 0] = 4.0
    dm[3, 1] = 5.0
    dm[3, 2] = 6.0

    buf = io.BytesIO()
    mpcf.save(dm, buf)
    buf.seek(0)
    dm2 = mpcf.load(buf)

    assert isinstance(dm2, mpcf.DistanceMatrix)
    assert dm2.size == 4
    np.testing.assert_allclose(dm.to_dense(), dm2.to_dense())


def test_save_load_empty_barcode_tensor():
    """Roundtrip a barcode tensor where all barcodes are empty."""
    bcs = mpcf.zeros((3,), dtype=mpcf.barcode64)
    buf = io.BytesIO()
    mpcf.save(bcs, buf)
    buf.seek(0)
    bcs2 = mpcf.load(buf)
    assert bcs2.shape == (3,)
    for i in range(3):
        assert len(bcs2[i]) == 0


def test_multiple_save_load_same_buffer():
    """Saving two objects to a buffer and loading back the first works."""
    f = mpcf.Pcf(np.array([[0.0, 5.0], [1.0, 0.0]]))

    buf = io.BytesIO()
    mpcf.save(f, buf)
    buf.seek(0)
    g = mpcf.load(buf)
    assert f == g
