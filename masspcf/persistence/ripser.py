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
from . import BarcodeTensor
from ..tensor import (FloatTensor, DoubleTensor, PointCloud32Tensor, PointCloud64Tensor,
                      _get_backend, zeros
                      )

from ..typing import f32, f64, pcloud32, pcloud64
from .tensor import Barcode32Tensor, Barcode64Tensor

from .._mpcf_cpp import persistence as cpp_p

def compute_barcodes_euclidean_pcloud_ripser(
        X : PointCloud32Tensor | PointCloud64Tensor | FloatTensor | DoubleTensor,
        maxDim : int = 1):

    backend = _get_backend(X, {
        f32 : cpp_p.PersistenceRipser32,
        f64 : cpp_p.PersistenceRipser64
    })

    if isinstance(X, FloatTensor):
        pcX = zeros((1,), dtype=pcloud32)
        pcX[0] = X
        X = PointCloud32Tensor(pcX)
    elif isinstance(X, DoubleTensor):
        pcX = zeros((1,), dtype=pcloud64)
        pcX[0] = X
        X = PointCloud64Tensor(pcX)

    out = backend.compute_barcodes_euclidean_pcloud_ripser(X._data, maxDim)

    if isinstance(X, PointCloud32Tensor):
        out = Barcode32Tensor(out)
    elif isinstance(X, PointCloud64Tensor):
        out = Barcode64Tensor(out)
    else:
        raise TypeError(f'Internal type error, please report this: Unhandled {type(X)}')

    if len(out.shape) == 2 and out.shape[0] == 1:
        out = out[0, :]

    return out

