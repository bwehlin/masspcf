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
from ..tensor import PointCloud32Tensor, PointCloud64Tensor, _get_backend
from ..typing import pcloud32, pcloud64
from .ph_tensor import Barcode32Tensor, Barcode64Tensor

cpp_p = cpp.persistence


def _compute_barcodes_euclidean_pcloud_ripser(
    X: PointCloud32Tensor | PointCloud64Tensor,
    out: Barcode32Tensor | Barcode64Tensor,
    maxDim: int = 1,
):
    backend, X = _get_backend(
        X, {pcloud32: cpp_p.PersistenceRipser32, pcloud64: cpp_p.PersistenceRipser64}
    )

    return backend.spawn_ripser_pcloud_euclidean_task(X._data, out._data, maxDim)
