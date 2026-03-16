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

from .. import _mpcf_cpp as cpp

cpp_p = cpp.persistence

from enum import Enum

import numpy as np

from ..async_task import _wait_for_task
from ..tensor import (
    Float32Tensor,
    Float64Tensor,
    PointCloud32Tensor,
    PointCloud64Tensor,
)
from ..typing import barcode32, barcode64, pcloud32, pcloud64
from .ph_tensor import Barcode32Tensor, Barcode64Tensor


class DistanceType(Enum):
    Euclidean = 1


class ComplexType(Enum):
    VietorisRips = 1


def compute_persistent_homology(
    X: PointCloud32Tensor
    | PointCloud64Tensor
    | Float32Tensor
    | Float64Tensor
    | np.ndarray,
    maxDim: int = 1,
    distance_type: DistanceType = DistanceType.Euclidean,
    complex_type: ComplexType = ComplexType.VietorisRips,
    verbose: bool = True,
) -> Barcode32Tensor | Barcode64Tensor:
    r"""Compute persistent homology of a point cloud.

    Returns barcodes for homology dimensions 0 through ``maxDim``. When
    ``complex_type`` is ``ComplexType.VietorisRips`` (currently the only
    available option), the computation uses Ripser [1]_. When the input
    contains multiple point clouds, the computations are parallelized
    across them.

    For an input tensor of shape :math:`(d_1, \ldots, d_n)`, the output has
    shape :math:`(d_1, \ldots, d_n, \texttt{maxDim} + 1)`.

    Parameters
    ----------
    X : PointCloud32Tensor, PointCloud64Tensor, Float32Tensor, Float64Tensor, or numpy.ndarray
        Input point cloud. A ``Float32/64Tensor`` or NumPy array is
        interpreted as a single point cloud (one row per point).
    maxDim : int, optional
        Maximum homology dimension to compute, by default 1.
    distance_type : DistanceType, optional
        Distance metric, by default ``DistanceType.Euclidean``.
    complex_type : ComplexType, optional
        Simplicial complex type, by default ``ComplexType.VietorisRips``.
    verbose : bool, optional
        Show progress information, by default True.

    Returns
    -------
    Barcode32Tensor or Barcode64Tensor
        A tensor of barcodes.

    References
    ----------
    .. [1] U. Bauer, "Ripser: efficient computation of Vietoris-Rips
       persistence barcodes", *Journal of Applied and Computational
       Topology*, vol. 5, pp. 391--423, 2021.
    """

    from ..tensor_create import zeros
    from .ripser import _compute_barcodes_euclidean_pcloud_ripser

    if isinstance(X, np.ndarray):
        if X.dtype == np.float32:
            X = Float32Tensor(X)
        elif X.dtype == np.float64:
            X = Float64Tensor(X)
        else:
            raise TypeError(
                f"Input has unsupported numpy dtype {X.dtype} (only np.float32/64 are supported)."
            )

    out = None
    if isinstance(X, Float32Tensor):
        pcX = zeros((1,), dtype=pcloud32)
        pcX[0] = X
        X = PointCloud32Tensor(pcX)
    elif isinstance(X, Float64Tensor):
        pcX = zeros((1,), dtype=pcloud64)
        pcX[0] = X
        X = PointCloud64Tensor(pcX)

    if isinstance(X, PointCloud32Tensor):
        out = zeros((1,), dtype=barcode32)
    elif isinstance(X, PointCloud64Tensor):
        out = zeros((1,), dtype=barcode64)

    if complex_type == ComplexType.VietorisRips:
        # Use Ripser
        match distance_type:
            case DistanceType.Euclidean:
                task = _compute_barcodes_euclidean_pcloud_ripser(X, out, maxDim)
            case _:
                raise ValueError(
                    f"Distance type {distance_type} not supported for complex type {complex_type}."
                )
    else:
        raise ValueError(f"Unsupported complex type {complex_type}")

    try:
        _wait_for_task(task, verbose)
    finally:
        if task is not None:
            task.request_stop()
            _wait_for_task(task, verbose=verbose)

    if len(out.shape) == 2 and out.shape[0] == 1:
        out = out[0, :]

    return out


"""
    backend, fs = _get_norms_backend(fs)
    out = np.zeros(X.shape, dtype=numpy_type(X))

    task = None
    try:
        task = backend.lpnorm_l1(out, X._data)
        _wait_for_task(task, verbose=verbose)
        return out
    finally:
        if task is not None:
            task.request_stop()
            _wait_for_task(task, verbose=verbose)
"""
