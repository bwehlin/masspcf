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

from . import _mpcf_cpp as cpp
from ._tensor_base import Shape, ShapeLike
from .tensor import (
    Float32Tensor,
    Float64Tensor,
    Pcf32iTensor,
    Pcf32Tensor,
    Pcf64iTensor,
    Pcf64Tensor,
    PointCloud32Tensor,
    PointCloud64Tensor,
)
from .typing import (
    _assert_valid_dtype,
    _check_deprecated_dtype,
    barcode32,
    barcode64,
    distmat32,
    distmat64,
    f32,
    f64,
    float32,  # Deprecated types
    float64,
    pcf32,
    pcf32i,
    pcf64,
    pcf64i,
    pcloud32,
    pcloud64,
    symmat32,
    symmat64,
)

cpp_p = cpp.persistence


def zeros(shape: ShapeLike, dtype=pcf32):
    """
    Creates a new `Tensor` of the specified `shape` and `dtype` whose entries are "zero." What "zero" means depends on the `dtype`:

    `dtype=pcf32/64`: A PCF that takes the value 0 for all times.
    `dtype=pcf32i/64i`: An integer PCF that takes the value 0 for all times.
    `dtype=f32/f64`: The number 0.
    `dtype=pcloud32/64`: An empty point cloud.
    `dtype=barcode32/64`: An empty barcode.
    `dtype=symmat32/64`: A 0×0 symmetric matrix.
    `dtype=distmat32/64`: A 0×0 distance matrix.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the returned tensor
    dtype
        The data type of the elements

    Returns
    -------
    Tensor
        The newly created tensor
    """

    from .persistence.ph_tensor import Barcode32Tensor, Barcode64Tensor

    if not isinstance(shape, Shape):
        shape = Shape(shape)  # If passed as, e.g., tuple of ints

    _check_deprecated_dtype(dtype)
    _assert_valid_dtype(
        dtype,
        [
            pcf32,
            pcf64,
            pcf32i,
            pcf64i,
            f32,
            f64,
            pcloud32,
            pcloud64,
            float32,
            float64,
            barcode32,
            barcode64,
            distmat32,
            distmat64,
            symmat32,
            symmat64,
        ],
    )

    if dtype in (pcf32, float32):
        return Pcf32Tensor(cpp.Pcf32Tensor(shape))
    elif dtype in (pcf64, float64):
        return Pcf64Tensor(cpp.Pcf64Tensor(shape))
    elif dtype == pcf32i:
        return Pcf32iTensor(cpp.Pcf32iTensor(shape))
    elif dtype == pcf64i:
        return Pcf64iTensor(cpp.Pcf64iTensor(shape))
    elif dtype == f32:
        return Float32Tensor(cpp.Float32Tensor(shape, 0.0))
    elif dtype == f64:
        return Float64Tensor(cpp.Float64Tensor(shape, 0.0))
    elif dtype == pcloud32:
        return PointCloud32Tensor(cpp.PointCloud32Tensor(shape))
    elif dtype == pcloud64:
        return PointCloud64Tensor(cpp.PointCloud64Tensor(shape))
    elif dtype == barcode32:
        return Barcode32Tensor(cpp_p.Barcode32Tensor(shape))
    elif dtype == barcode64:
        return Barcode64Tensor(cpp_p.Barcode64Tensor(shape))
    elif dtype == distmat32:
        from .distance_matrix import DistanceMatrix32Tensor
        return DistanceMatrix32Tensor(cpp.DistanceMatrix32Tensor(shape))
    elif dtype == distmat64:
        from .distance_matrix import DistanceMatrix64Tensor
        return DistanceMatrix64Tensor(cpp.DistanceMatrix64Tensor(shape))
    elif dtype == symmat32:
        from .symmetric_matrix import SymmetricMatrix32Tensor
        return SymmetricMatrix32Tensor(cpp.SymmetricMatrix32Tensor(shape))
    elif dtype == symmat64:
        from .symmetric_matrix import SymmetricMatrix64Tensor
        return SymmetricMatrix64Tensor(cpp.SymmetricMatrix64Tensor(shape))
    else:
        raise NotImplementedError("This dtype has not been implemented.")
