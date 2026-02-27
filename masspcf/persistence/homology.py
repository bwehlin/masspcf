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

from ..async_task import _wait_for_task

from ..tensor import (FloatTensor, DoubleTensor, PointCloud32Tensor, PointCloud64Tensor,
                      _get_backend
                      )

from ..typing import pcloud32, pcloud64, barcode32, barcode64

from .ripser import compute_barcodes_euclidean_pcloud_ripser

from .ph_tensor import Barcode32Tensor, Barcode64Tensor

from enum import Enum
import numpy as np

class DistanceType(Enum):
    Euclidean = 1

class ComplexType(Enum):
    VietorisRips = 1

def compute_persistent_homology(X : PointCloud32Tensor | PointCloud64Tensor | FloatTensor | DoubleTensor | np.ndarray,
                                distance_type : DistanceType = DistanceType.Euclidean,
                                complex_type: ComplexType = ComplexType.VietorisRips ) \
        -> Barcode32Tensor | Barcode64Tensor :

    from ..tensor_create import zeros

    if isinstance(X, np.ndarray):
        if X.dtype == np.float32:
            X = FloatTensor(X)
        elif X.dtype == np.float64:
            X = DoubleTensor(X)
        else:
            raise TypeError(f'Input has unsupported numpy dtype {X.dtype} (only np.float32/64 are supported).')

    if isinstance(X, FloatTensor):
        pcX = zeros((1,), dtype=pcloud32)
        pcX[0] = X
        X = PointCloud32Tensor(pcX)
        out = zeros((1,), dtype=barcode32)
    elif isinstance(X, DoubleTensor):
        pcX = zeros((1,), dtype=pcloud64)
        pcX[0] = X
        X = PointCloud64Tensor(pcX)
        out = zeros((1,), dtype=barcode64)

    if complex_type == ComplexType.VietorisRips:
        # Use Ripser
        match distance_type:
            case DistanceType.Euclidean:
                task = compute_barcodes_euclidean_pcloud_ripser()
            case _:
                raise ValueError(f'Distance type {distance_type} not supported for complex type {complex_type}.')
    else:
        raise ValueError(f'Unsupported complex type {complex_type}')

    if len(out.shape) == 2 and out.shape[0] == 1:
        out = out[0, :]


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
