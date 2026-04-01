"""Tests for masspcf.system configuration functions."""

import pytest

import masspcf as mpcf
from masspcf import _mpcf_cpp as cpp


def test_force_cpu_enables_and_disables():
    """force_cpu(True/False) should not raise."""
    mpcf.system.force_cpu(True)
    mpcf.system.force_cpu(False)


def test_force_cpu_affects_computation():
    """Computation should succeed under force_cpu(True)."""
    mpcf.system.force_cpu(True)
    try:
        X = mpcf.random.noisy_sin((5,), dtype=mpcf.pcf64)
        dm = mpcf.pdist(X, verbose=False)
        assert dm.size == 5
    finally:
        mpcf.system.force_cpu(False)


def test_limit_cpus():
    """limit_cpus should not raise for reasonable values."""
    mpcf.system.limit_cpus(1)
    # Verify computation still works with 1 CPU
    X = mpcf.random.noisy_sin((4,), dtype=mpcf.pcf64)
    dm = mpcf.pdist(X, verbose=False)
    assert dm.size == 4
    # Restore to a reasonable default
    mpcf.system.limit_cpus(4)


_is_cuda_build = cpp._build_type() == "CUDA"


@pytest.mark.skipif(not _is_cuda_build, reason="GPU functions only available in CUDA build")
def test_limit_gpus():
    """limit_gpus should not raise."""
    mpcf.system.limit_gpus(0)
    mpcf.system.limit_gpus(1)


@pytest.mark.skipif(not _is_cuda_build, reason="GPU functions only available in CUDA build")
def test_set_cuda_threshold():
    """set_cuda_threshold should not raise."""
    mpcf.system.set_cuda_threshold(100)
    mpcf.system.set_cuda_threshold(500)  # restore default


@pytest.mark.skipif(not _is_cuda_build, reason="GPU functions only available in CUDA build")
def test_set_device_verbose():
    """set_device_verbose on and off should not raise."""
    mpcf.system.set_device_verbose(True)
    mpcf.system.set_device_verbose(False)


@pytest.mark.skipif(not _is_cuda_build, reason="GPU functions only available in CUDA build")
def test_set_block_size():
    """set_block_size should not raise for valid dimensions."""
    mpcf.system.set_block_size(16, 16)
    mpcf.system.set_block_size(32, 32)


@pytest.mark.skipif(not _is_cuda_build, reason="GPU functions only available in CUDA build")
def test_set_min_block_side():
    """set_min_block_side should not raise."""
    mpcf.system.set_min_block_side(0)  # auto-detect
    mpcf.system.set_min_block_side(64)
    mpcf.system.set_min_block_side(0)  # restore


def test_build_type_returns_string():
    """build_type should return a non-empty string."""
    bt = mpcf.system.build_type()
    assert isinstance(bt, str)
    assert len(bt) > 0
    assert bt in ("CPU", "CUDA")
