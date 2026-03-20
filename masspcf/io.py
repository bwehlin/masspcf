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

import masspcf._mpcf_cpp as cpp

from .persistence.ph_tensor import BarcodeTensor
from .distance_matrix import DistanceMatrixTensor
from .symmetric_matrix import SymmetricMatrixTensor
from .tensor import (
    FloatTensor,
    IntPcfTensor,
    PcfTensor,
    PointCloudTensor,
    Tensor,
)
from .typing import (
    barcode32,
    barcode64,
    distmat32,
    distmat64,
    float32,
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


def _save(item: Tensor, file):
    _SAVE_DISPATCH = {
        float32: cpp.IoOps.save_float32_tensor,
        float64: cpp.IoOps.save_float64_tensor,
        pcf32: cpp.IoOps.save_pcf32_tensor,
        pcf64: cpp.IoOps.save_pcf64_tensor,
        pcf32i: cpp.IoOps.save_pcf32i_tensor,
        pcf64i: cpp.IoOps.save_pcf64i_tensor,
        pcloud32: cpp.IoOps.save_point_cloud32_tensor,
        pcloud64: cpp.IoOps.save_point_cloud64_tensor,
        barcode32: cpp.IoOps.save_barcode32_tensor,
        barcode64: cpp.IoOps.save_barcode64_tensor,
        symmat32: cpp.IoOps.save_symmetric_matrix32_tensor,
        symmat64: cpp.IoOps.save_symmetric_matrix64_tensor,
        distmat32: cpp.IoOps.save_distance_matrix32_tensor,
        distmat64: cpp.IoOps.save_distance_matrix64_tensor,
    }

    fn = _SAVE_DISPATCH.get(item.dtype)
    if fn is None:
        raise TypeError(f"Unsupported tensor dtype {item.dtype}")
    fn(item._data, file)


def _load(file):
    cpp_p = cpp.persistence

    _LOAD_DISPATCH = {
        cpp.Float32Tensor: FloatTensor,
        cpp.Float64Tensor: FloatTensor,
        cpp.Pcf32Tensor: PcfTensor,
        cpp.Pcf64Tensor: PcfTensor,
        cpp.Pcf32iTensor: IntPcfTensor,
        cpp.Pcf64iTensor: IntPcfTensor,
        cpp.PointCloud32Tensor: PointCloudTensor,
        cpp.PointCloud64Tensor: PointCloudTensor,
        cpp_p.Barcode32Tensor: BarcodeTensor,
        cpp_p.Barcode64Tensor: BarcodeTensor,
        cpp.SymmetricMatrix32Tensor: SymmetricMatrixTensor,
        cpp.SymmetricMatrix64Tensor: SymmetricMatrixTensor,
        cpp.DistanceMatrix32Tensor: DistanceMatrixTensor,
        cpp.DistanceMatrix64Tensor: DistanceMatrixTensor,
    }

    cpp_tensor = cpp.IoOps.load_tensor_from_file(file)
    ctor = _LOAD_DISPATCH.get(type(cpp_tensor))
    if ctor is None:
        raise TypeError(f"File contains unsupported tensor of type {type(cpp_tensor)}")
    return ctor(cpp_tensor)


def save(item: Tensor, file):
    """Save a tensor to a file in masspcf's binary format.

    All tensor types are supported.

    Parameters
    ----------
    item : Tensor
        The tensor to save.
    file : str or file-like
        A file path or an open file object in binary write mode.
    """
    if isinstance(file, str):
        with open(file, "wb") as f:
            _save(item, f)
    else:
        _save(item, file)


def load(file) -> Tensor:
    """Load a tensor from a file in masspcf's binary format.

    The returned tensor will have the same type and dtype as what was saved.

    Parameters
    ----------
    file : str or file-like
        A file path or an open file object in binary read mode.

    Returns
    -------
    Tensor
        The loaded tensor.
    """
    if isinstance(file, str):
        with open(file, "rb") as f:
            return _load(f)
    else:
        return _load(file)
