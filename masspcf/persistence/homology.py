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

from enum import Enum

import numpy as np

from .. import _mpcf_cpp as cpp
from ..async_task import _run_task
from ..distance_matrix import (
    DistanceMatrix,
    DistanceMatrixTensor,
)
from ..tensor import (
    FloatTensor,
    PointCloudTensor,
)
from ..typing import (
    barcode32, barcode64,
    distmat32, distmat64,
    float32, float64,
    pcloud32, pcloud64,
)
from .ph_tensor import BarcodeTensor

cpp_p = cpp.persistence

_DISTMAT_TO_BARCODE_DTYPE = {distmat32: barcode32, distmat64: barcode64}
_PCLOUD_TO_BARCODE_DTYPE = {pcloud32: barcode32, pcloud64: barcode64}
_FLOAT_TO_PCLOUD_DTYPE = {float32: pcloud32, float64: pcloud64}


class DistanceType(Enum):
    Euclidean = 1


class ComplexType(Enum):
    VietorisRips = 1


def compute_persistent_homology(
    X: PointCloudTensor
    | DistanceMatrix
    | DistanceMatrixTensor
    | FloatTensor
    | np.ndarray,
    max_dim: int = 1,
    distance_type: DistanceType = DistanceType.Euclidean,
    complex_type: ComplexType = ComplexType.VietorisRips,
    reduced: bool = False,
    verbose: bool = False,
) -> BarcodeTensor:
    r"""Compute persistent homology of a point cloud or distance matrix.

    Returns barcodes for homology dimensions 0 through ``max_dim``. When
    ``complex_type`` is ``ComplexType.VietorisRips`` (currently the only
    available option), the computation uses Ripser :footcite:`Bauer2021`. When the input
    contains multiple point clouds or distance matrices, the computations are
    parallelized across them.

    For an input tensor of shape :math:`(d_1, \ldots, d_n)`, the output has
    shape :math:`(d_1, \ldots, d_n, \texttt{max_dim} + 1)`.

    Parameters
    ----------
    X : PointCloudTensor, DistanceMatrix, DistanceMatrixTensor, FloatTensor, or numpy.ndarray
        Input data. A ``FloatTensor`` or NumPy array is
        interpreted as a single point cloud (one row per point).
        A ``DistanceMatrix`` or ``DistanceMatrixTensor`` provides
        precomputed pairwise distances directly; ``distance_type`` is
        ignored in that case.
    max_dim : int, optional
        Maximum homology dimension to compute, by default 1.
    distance_type : DistanceType, optional
        Distance metric for point cloud input, by default
        ``DistanceType.Euclidean``. Ignored when the input is a distance
        matrix.
    complex_type : ComplexType, optional
        Simplicial complex type, by default ``ComplexType.VietorisRips``.
    reduced : bool, optional
        If ``True``, compute reduced homology (no essential H0 class).
        If ``False`` (default), an infinite bar ``[0, inf)`` is added to
        H0 representing the single connected component.
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    BarcodeTensor
        A tensor of barcodes.

    """

    from ..tensor_create import zeros
    from ..timeseries import TensorTimeSeries
    from .ripser import _compute_barcodes_distmat_ripser, _compute_barcodes_euclidean_pcloud_ripser

    # --- TensorTimeSeries of point clouds -> TensorTimeSeries of barcode tensors ---
    if isinstance(X, TensorTimeSeries) and X.dtype in (float32, float64):
        if X.n_channels != 1:
            raise ValueError(
                "compute_persistent_homology only accepts single-channel "
                "TensorTimeSeries inputs")
        if complex_type != ComplexType.VietorisRips:
            raise ValueError(f"Unsupported complex type {complex_type}")
        if distance_type != DistanceType.Euclidean:
            raise ValueError(
                f"Distance type {distance_type} not supported for complex type {complex_type}.")

        clouds = X.values  # list of FloatTensor, one per timestep
        n_times = len(clouds)
        pcloud_dtype = pcloud32 if X.dtype is float32 else pcloud64
        barcode_dtype = _PCLOUD_TO_BARCODE_DTYPE[pcloud_dtype]

        # Bundle the per-timestep point clouds into a 1-D PointCloudTensor
        # so we can make a single batched Ripser call.
        pclouds = zeros((n_times,), dtype=pcloud_dtype)
        for i, pc in enumerate(clouds):
            pclouds[i] = pc

        out = zeros((n_times, max_dim + 1), dtype=barcode_dtype)
        task = _compute_barcodes_euclidean_pcloud_ripser(
            pclouds, out, max_dim, reduced)
        _run_task(lambda: task, verbose=verbose)

        # Split row-wise: each timestep owns a BarcodeTensor of shape (max_dim+1,).
        per_step = [out[i] for i in range(n_times)]
        return TensorTimeSeries(X.times, per_step)

    # --- Distance matrix input path ---
    if isinstance(X, (DistanceMatrix, DistanceMatrixTensor)):
        if isinstance(X, DistanceMatrix):
            # Wrap single DistanceMatrix into a 1-element tensor
            if isinstance(X._data, cpp.DistanceMatrix_f32):
                dmX = zeros((1,), dtype=distmat32)
            else:
                dmX = zeros((1,), dtype=distmat64)
            dmX[0] = X
            X = dmX

        barcode_dtype = _DISTMAT_TO_BARCODE_DTYPE[X.dtype]
        out = zeros((1,), dtype=barcode_dtype)

        if complex_type != ComplexType.VietorisRips:
            raise ValueError(f"Unsupported complex type {complex_type}")

        task = _compute_barcodes_distmat_ripser(X, out, max_dim, reduced)
        _run_task(lambda: task, verbose=verbose)

        if len(out.shape) == 2 and out.shape[0] == 1:
            out = out[0, :]

        return out

    # --- Point cloud input path ---
    if isinstance(X, np.ndarray):
        X = FloatTensor(X)

    out = None
    if isinstance(X, FloatTensor):
        pcloud_dtype = _FLOAT_TO_PCLOUD_DTYPE[X.dtype]
        pcX = zeros((1,), dtype=pcloud_dtype)
        pcX[0] = X
        X = pcX

    if isinstance(X, PointCloudTensor):
        barcode_dtype = _PCLOUD_TO_BARCODE_DTYPE[X.dtype]
        out = zeros((1,), dtype=barcode_dtype)

    if complex_type == ComplexType.VietorisRips:
        # Use Ripser
        match distance_type:
            case DistanceType.Euclidean:
                task = _compute_barcodes_euclidean_pcloud_ripser(X, out, max_dim, reduced)
            case _:
                raise ValueError(
                    f"Distance type {distance_type} not supported for complex type {complex_type}."
                )
    else:
        raise ValueError(f"Unsupported complex type {complex_type}")

    _run_task(lambda: task, verbose=verbose)

    if len(out.shape) == 2 and out.shape[0] == 1:
        out = out[0, :]

    return out
