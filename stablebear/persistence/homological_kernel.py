import numpy as np

from .. import _sb_cpp as cpp
from ..distance_matrix import (
    DistanceMatrix,
    DistanceMatrixTensor,
)
from ..base_tensor import (
    FloatTensor,
    PointCloudTensor,
)
from ..typing import barcode32, barcode64, distmat32, distmat64, float32, float64, pcloud32, pcloud64
from .homology import DistanceType
from .ph_tensor import BarcodeTensor

cpp_p = cpp.persistence

_DISTMAT_TO_BARCODE_DTYPE = {distmat32: barcode32, distmat64: barcode64}
_PCLOUD_TO_BARCODE_DTYPE = {pcloud32: barcode32, pcloud64: barcode64}
_FLOAT_TO_PCLOUD_DTYPE = {float32: pcloud32, float64: pcloud64}


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
    | np.ndarray,
    *,
    metric: DistanceType = DistanceType.Euclidean,
    verbose: bool = False,
) -> BarcodeTensor:
    r"""Compute the 0th homological kernel between two distance structures.

    ``X`` carries the larger distances :math:`d`; ``X_prime`` carries the
    dominated distances :math:`d'` (:math:`d' \le d` pointwise). Both inputs
    must be the same kind and the same shape; element ``i`` of ``X`` is
    paired with element ``i`` of ``X_prime``.

    Parameters
    ----------
    X : PointCloudTensor, DistanceMatrix, DistanceMatrixTensor, FloatTensor, or numpy.ndarray
        Input data for :math:`d`. A ``FloatTensor`` or NumPy array is
        interpreted as a single point cloud (one row per point).
        A ``DistanceMatrix`` or ``DistanceMatrixTensor`` provides
        precomputed pairwise distances directly; ``metric`` is ignored
        in that case.
    X_prime : same kind as ``X``
        Input data for :math:`d'`, with the same number of points per
        element as ``X`` (point cloud dimensions may differ).
    metric : DistanceType, optional
        Distance metric applied to both point cloud inputs, by default
        ``DistanceType.Euclidean``. Ignored when the inputs are distance
        matrices.
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    BarcodeTensor
        A tensor of kernel barcodes, one per input element, each with
        :math:`n - 1` finite bars.

    """
    raise NotImplementedError
