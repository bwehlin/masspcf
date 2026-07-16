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
    DistanceType,
)
from .ph_tensor import BarcodeTensor

cpp_p = cpp.persistence

_FLOAT_TO_DISTMAT_DTYPE = {float32: distmat32, float64: distmat64}


def _spawn_homological_kernel_pcloud_task(
    X: PointCloudTensor,
    X_prime: PointCloudTensor,
    out: BarcodeTensor,
):
    backend, X = _get_backend(
        X, {pcloud32: cpp_p.HomologicalKernel32, pcloud64: cpp_p.HomologicalKernel64}
    )

    return backend.spawn_homological_kernel_pcloud_task(X._data, X_prime._data, out._data)


def _spawn_homological_kernel_distmat_task(
    X: DistanceMatrixTensor,
    X_prime: DistanceMatrixTensor,
    out: BarcodeTensor,
):
    backend, X = _get_backend(
        X, {distmat32: cpp_p.HomologicalKernel32, distmat64: cpp_p.HomologicalKernel64}
    )

    return backend.spawn_homological_kernel_distmat_task(X._data, X_prime._data, out._data)


def _project_diagonal(points: np.ndarray) -> np.ndarray:
    """Orthogonal projection onto the diagonal line spanned by (1, ..., 1).

    Each cloud is centered first and the offset is not restored (distances,
    which are all the kernel consumes, are translation invariant). The
    projected coordinates are stored in the cloud's dtype, so their rounding
    error scales with coordinate magnitude; centering keeps that magnitude at
    the cloud's spread instead of its distance from the origin.
    """
    if points.shape[-2] == 0:
        return points.copy()
    centered = points - points.mean(axis=-2, keepdims=True)
    return np.broadcast_to(centered.mean(axis=-1, keepdims=True), points.shape).copy()


def _project_coordinate(points: np.ndarray) -> np.ndarray:
    """Orthogonal projection onto the first dim-1 coordinate axes."""
    out = points.copy()
    out[..., -1] = 0
    return out


# Certified transform presets: every entry is an orthogonal projection, hence
# 1-Lipschitz with respect to the Euclidean metric in every dimension, so the
# resulting d' is guaranteed to be dominated by d. Each maps an (n, dim) array
# to an (n, dim) array (projections stay in ambient coordinates) and operates
# on the trailing axes only, so a stacked (..., n, dim) batch projects in one
# call.
_TRANSFORM_PRESETS = {
    "diagonal": _project_diagonal,
    "coordinate": _project_coordinate,
}


def _apply_transform(X: PointCloudTensor, transform: str) -> PointCloudTensor:
    from ..tensor_create import zeros

    preset = _TRANSFORM_PRESETS.get(transform)
    if preset is None:
        raise ValueError(
            f"Unknown transform {transform!r}; available presets: {sorted(_TRANSFORM_PRESETS)}"
        )

    shape = tuple(X.shape)
    # _get_element directly: __getitem__'s fancy-index parsing costs ~6x per
    # element, which dominates for large batches of small clouds.
    data = X._data
    clouds = [np.array(data._get_element(list(idx)), copy=False) for idx in np.ndindex(shape)]
    if not clouds:
        return zeros(shape, dtype=X.dtype)

    if all(c.shape == clouds[0].shape for c in clouds):
        # Uniform batch: one vectorized projection, one tensor construction.
        stacked = np.stack(clouds).reshape(shape + clouds[0].shape)
        return PointCloudTensor(preset(stacked), cloud_ndim=clouds[0].ndim, dtype=X.dtype)

    projected = [preset(c) for c in clouds]
    if len(shape) == 1:
        return PointCloudTensor(projected, dtype=X.dtype)

    X_prime = zeros(shape, dtype=X.dtype)
    for idx, cloud in zip(np.ndindex(shape), projected):
        X_prime[idx] = cloud
    return X_prime


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
    X_prime: PointCloudTensor
    | DistanceMatrix
    | DistanceMatrixTensor
    | FloatTensor
    | np.ndarray
    | None = None,
    *,
    transform: str | None = None,
    metric: DistanceType = DistanceType.Euclidean,
    verbose: bool = False,
) -> BarcodeTensor:
    r"""Compute the 0th homological kernel between two distance structures.

    ``X`` carries the larger distances :math:`d`; the dominated distances
    :math:`d'` (:math:`d' \le d` pointwise) come from exactly one of two
    sources: an explicit ``X_prime``, or a built-in ``transform`` preset
    applied to ``X``. When ``X_prime`` is given it must be the same kind
    and the same shape as ``X``; element ``i`` of ``X`` is paired with
    element ``i`` of ``X_prime``. When the input contains multiple point
    clouds or distance matrices, the computations are parallelized across
    them.

    Parameters
    ----------
    X : PointCloudTensor, DistanceMatrix, DistanceMatrixTensor, FloatTensor, or numpy.ndarray
        Input data for :math:`d`. A ``FloatTensor`` or NumPy array is
        interpreted as a single point cloud (one row per point).
        A ``DistanceMatrix`` or ``DistanceMatrixTensor`` provides
        precomputed pairwise distances directly; ``metric`` is ignored
        in that case.
    X_prime : same kind as ``X``, optional
        Input data for :math:`d'`, with the same number of points per
        element as ``X``. Point clouds stay in ambient dimension: express
        a lower-dimensional projection in the original coordinates.
        Mutually exclusive with ``transform``.
    transform : str, optional
        Name of a built-in transform applied to each point cloud in ``X``
        to produce :math:`d'`. Every preset is an orthogonal projection,
        hence valid for any point cloud in any dimension. Mutually
        exclusive with ``X_prime``; requires point cloud input. Available
        presets:

        - ``"diagonal"``: projection onto the diagonal line spanned by
          :math:`(1, \ldots, 1)`.
        - ``"coordinate"``: projection onto the first :math:`\dim - 1`
          coordinate axes (drops the last coordinate).
    metric : DistanceType, optional
        Distance metric applied to both point cloud inputs, by default
        ``DistanceType.Euclidean``. Ignored when the inputs are distance
        matrices.
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
        If ``X`` and ``X_prime`` are not the same kind or not the same
        dtype, or if ``transform`` is used with distance matrix input.
    ValueError
        If neither or both of ``X_prime`` and ``transform`` are given,
        the (outer) tensor shapes differ, the transform is unknown, or
        the metric is unsupported.
    RuntimeError
        If :math:`d` does not dominate :math:`d'` (a death would land
        below its birth by more than floating-point roundoff), or if
        paired elements mismatch (for example, two point clouds with
        different point counts, or a point cloud that is not an
        ``(n, dim)`` array).

    """
    from ..tensor_create import zeros

    if (X_prime is None) == (transform is None):
        raise ValueError("Provide exactly one of X_prime or transform")

    X = _normalize_kernel_input(X, "X")

    if transform is not None:
        if not isinstance(X, PointCloudTensor):
            raise TypeError("transform presets require point cloud input")
        X_prime = _apply_transform(X, transform)
    else:
        X_prime = _normalize_kernel_input(X_prime, "X_prime")

    if isinstance(X, PointCloudTensor) != isinstance(X_prime, PointCloudTensor):
        raise TypeError(
            "X and X_prime must be the same kind "
            f"(got {type(X).__name__} and {type(X_prime).__name__})"
        )
    if X.dtype != X_prime.dtype:
        raise TypeError(f"X and X_prime must have the same dtype (got {X.dtype} and {X_prime.dtype})")
    if X.shape != X_prime.shape:
        raise ValueError(f"X and X_prime must have the same shape (got {X.shape} and {X_prime.shape})")

    if isinstance(X, PointCloudTensor):
        if metric != DistanceType.Euclidean:
            raise ValueError(f"Unsupported metric {metric}")
        out = zeros((1,), dtype=_PCLOUD_TO_BARCODE_DTYPE[X.dtype])
        task = _spawn_homological_kernel_pcloud_task(X, X_prime, out)
    else:
        out = zeros((1,), dtype=_DISTMAT_TO_BARCODE_DTYPE[X.dtype])
        task = _spawn_homological_kernel_distmat_task(X, X_prime, out)

    _run_task(lambda: task, verbose=verbose)

    return out
