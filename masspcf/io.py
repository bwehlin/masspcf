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

from .tensor import Tensor, Float32Tensor, Float64Tensor, Pcf32Tensor, Pcf64Tensor, PointCloud32Tensor, PointCloud64Tensor
from .persistence import Barcode32Tensor, Barcode64Tensor

import masspcf._mpcf_cpp as cpp

def _save(item: Tensor, file):
    _SAVE_DISPATCH = {
        Float32Tensor:      cpp.IoOps.save_float32_tensor,
        Float64Tensor:      cpp.IoOps.save_float64_tensor,
        Pcf32Tensor:        cpp.IoOps.save_pcf32_tensor,
        Pcf64Tensor:        cpp.IoOps.save_pcf64_tensor,
        PointCloud32Tensor: cpp.IoOps.save_point_cloud32_tensor,
        PointCloud64Tensor: cpp.IoOps.save_point_cloud64_tensor,
        Barcode32Tensor:    cpp.IoOps.save_barcode32_tensor,
        Barcode64Tensor:    cpp.IoOps.save_barcode64_tensor,
    }

    fn = _SAVE_DISPATCH.get(type(item))
    if fn is None:
        raise TypeError(f'Unsupported tensor type {type(item)}')
    fn(item._data, file)

def _load(file):
    cpp_p = cpp.persistence

    _LOAD_DISPATCH = {
        cpp.Float32Tensor:      Float32Tensor,
        cpp.Float64Tensor:      Float64Tensor,
        cpp.Pcf32Tensor:        Pcf32Tensor,
        cpp.Pcf64Tensor:        Pcf64Tensor,
        cpp.PointCloud32Tensor: PointCloud32Tensor,
        cpp.PointCloud64Tensor: PointCloud64Tensor,
        cpp_p.Barcode32Tensor:  Barcode32Tensor,
        cpp_p.Barcode64Tensor:  Barcode64Tensor,
    }

    cpp_tensor = cpp.IoOps.load_tensor_from_file(file)
    ctor = _LOAD_DISPATCH.get(type(cpp_tensor))
    if ctor is None:
        raise TypeError(f'File contains unsupported tensor of type {type(cpp_tensor)}')
    return ctor(cpp_tensor)

def save(item: Tensor, file):
    """Save a tensor to a file in masspcf's binary format.

    All tensor types are supported (PCF, numeric, point cloud, barcode).

    Parameters
    ----------
    item : Tensor
        The tensor to save.
    file : str or file-like
        A file path or an open file object in binary write mode.
    """
    if isinstance(file, str):
        with open(file, 'wb') as f:
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
        with open(file, 'rb') as f:
            return _load(f)
    else:
        return _load(file)

