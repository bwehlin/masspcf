import numpy as np

from .. import _sb_cpp as cpp
from ..async_task import _run_task
from ..distance_matrix import (
    DistanceMatrix,
    DistanceMatrixTensor,
)
from ..base_tensor import (
    FloatTensor,
    PointCloudTensor,
    _get_backend,
)
from ..typing import distmat32, distmat64, float32, float64, pcloud32, pcloud64
from .homology import (
    _DISTMAT_TO_BARCODE_DTYPE,
    _FLOAT_TO_PCLOUD_DTYPE,
    _PCLOUD_TO_BARCODE_DTYPE,
)
from .ph_tensor import BarcodeTensor

cpp_p = cpp.persistence

_FLOAT_TO_DISTMAT_DTYPE = {float32: distmat32, float64: distmat64}


def _spawn_homological_kernel_pcloud_task(
    X: PointCloudTensor,
    Y: PointCloudTensor,
    out: BarcodeTensor,
):
    backend, X = _get_backend(
        X, {pcloud32: cpp_p.HomologicalKernel32, pcloud64: cpp_p.HomologicalKernel64}
    )

    return backend.spawn_homological_kernel_pcloud_task(X._data, Y._data, out._data)


def _spawn_homological_kernel_distmat_task(
    X: DistanceMatrixTensor,
    Y: DistanceMatrixTensor,
    out: BarcodeTensor,
):
    backend, X = _get_backend(
        X, {distmat32: cpp_p.HomologicalKernel32, distmat64: cpp_p.HomologicalKernel64}
    )

    return backend.spawn_homological_kernel_distmat_task(X._data, Y._data, out._data)


def _normalize_kernel_input(X, name: str):
    """Normalize one kernel input to a tensor, wrapping single instances
    into a 1-element tensor."""
    from ..tensor_create import zeros

    if isinstance(X, np.ndarray):
        X = FloatTensor(X)

    if isinstance(X, FloatTensor):
        pcX = zeros((1,), dtype=_FLOAT_TO_PCLOUD_DTYPE[X.dtype])
        pcX[0] = X
        return pcX

    if isinstance(X, DistanceMatrix):
        dmX = zeros((1,), dtype=_FLOAT_TO_DISTMAT_DTYPE[X.dtype])
        dmX[0] = X
        return dmX

    if isinstance(X, (PointCloudTensor, DistanceMatrixTensor)):
        return X

    raise TypeError(f"compute_homological_kernel does not support {name} of type {type(X)}")


def compute_homological_kernel(
    X: PointCloudTensor
    | DistanceMatrix
    | DistanceMatrixTensor
    | FloatTensor
    | np.ndarray,
    Y: PointCloudTensor
    | DistanceMatrix
    | DistanceMatrixTensor
    | FloatTensor
    | np.ndarray,
    *,
    dim: int = 0,
    verbose: bool = False,
) -> BarcodeTensor:
    r"""Compute the homological kernel between two distance structures.

    ``X`` carries the larger distances :math:`d` and ``Y`` the dominated
    distances :math:`d'` (:math:`d' \le d` pointwise). Both must be the
    same kind and the same shape; element ``i`` of ``X`` is paired with
    element ``i`` of ``Y``. Point clouds always use the Euclidean metric;
    for any other metric, pass precomputed distance matrices. When the
    input contains multiple point clouds or distance matrices, the
    computations are parallelized across them.

    Parameters
    ----------
    X : PointCloudTensor, DistanceMatrix, DistanceMatrixTensor, FloatTensor, or numpy.ndarray
        Input data for :math:`d`. A ``FloatTensor`` or NumPy array is
        interpreted as a single point cloud (one row per point).
    Y : same kind as ``X``
        Input data for :math:`d'`, with the same number of points per
        element as ``X``. Point clouds stay in ambient dimension: express
        a lower-dimensional projection in the original coordinates.
    dim : int, optional
        Homology dimension of the kernel, by default 0. Only ``dim=0``
        is currently supported.
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    BarcodeTensor
        Kernel barcodes with the same shape as the input, one per input
        element, each with :math:`n - 1` finite bars. Single-instance
        inputs (a NumPy array, ``FloatTensor``, or ``DistanceMatrix``)
        yield a tensor of shape ``(1,)``.

    Raises
    ------
    TypeError
        If ``X`` and ``Y`` are not the same kind or not the same dtype.
    ValueError
        If the (outer) tensor shapes differ, or ``dim`` is negative.
    NotImplementedError
        If ``dim`` is positive.
    RuntimeError
        If :math:`d` does not dominate :math:`d'` (a death would land
        below its birth by more than floating-point roundoff), or if
        paired elements mismatch (for example, two point clouds with
        different point counts, or a point cloud that is not an
        ``(n, dim)`` array).

    """
    from ..tensor_create import zeros

    if dim < 0:
        raise ValueError(f"dim must be non-negative (got {dim})")
    if dim > 0:
        raise NotImplementedError(f"only dim=0 is currently supported (got dim={dim})")

    X = _normalize_kernel_input(X, "X")
    Y = _normalize_kernel_input(Y, "Y")

    if isinstance(X, PointCloudTensor) != isinstance(Y, PointCloudTensor):
        raise TypeError(
            "X and Y must be the same kind "
            f"(got {type(X).__name__} and {type(Y).__name__})"
        )
    if X.dtype != Y.dtype:
        raise TypeError(f"X and Y must have the same dtype (got {X.dtype} and {Y.dtype})")
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape (got {X.shape} and {Y.shape})")

    if isinstance(X, PointCloudTensor):
        out = zeros((1,), dtype=_PCLOUD_TO_BARCODE_DTYPE[X.dtype])
        task = _spawn_homological_kernel_pcloud_task(X, Y, out)
    else:
        out = zeros((1,), dtype=_DISTMAT_TO_BARCODE_DTYPE[X.dtype])
        task = _spawn_homological_kernel_distmat_task(X, Y, out)

    _run_task(lambda: task, verbose=verbose)

    return out
