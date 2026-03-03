#    Copyright 2024-2026 Bjorn Wehlin
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import annotations

from . import _mpcf_cpp as cpp
from ._tensor_base import Tensor

from .typing import (pcf32, pcf64, f32, f64, pcloud32, pcloud64, _validate_dtype)

from .pcf import Pcf

from typing import Union
import numpy as np

class NumericTensor(Tensor):
    def __init__(self):
        super().__init__()

    def _get_valid_setitem_dtypes(self):
        return [NumericTensor, Float32Tensor, Float64Tensor, float, int, np.ndarray]

    def _decay_value(self, val):
        return val

    def _represent_element(self, element):
        return element

    def __array__(self):
        return np.array(self._data)

class Float32Tensor(NumericTensor):
    def __init__(self, data : cpp.Float32Tensor | Float32Tensor | np.ndarray):
        super().__init__()

        if isinstance(data, cpp.Float32Tensor):
            pass
        elif isinstance(data, Float32Tensor):
            data = data._data
        elif isinstance(data, np.ndarray):
            data = cpp.ndarray_to_tensor_32(data)
        else:
            raise TypeError(f'Cannot create {type(self)} from {type(data)}')

        self._data = data
        self.dtype = f32

    def _to_py_tensor(self, data):
        return Float32Tensor(data)

class Float64Tensor(NumericTensor):
    def __init__(self, data : cpp.Float64Tensor | Float64Tensor | np.ndarray):
        super().__init__()

        if isinstance(data, cpp.Float64Tensor):
            pass
        elif isinstance(data, Float64Tensor):
            data = data._data
        elif isinstance(data, np.ndarray):
            data = cpp.ndarray_to_tensor_64(data)
        else:
            raise TypeError(f'Cannot create {type(self)} from {type(data)}')

        self._data = data
        self.dtype = f64

    def _to_py_tensor(self, data):
        return Float64Tensor(data)

    def __repr__(self):
        return np.asarray(self).__repr__()

    def __str__(self):
        return np.asarray(self).__str__()

class PcfTensor(Tensor):
    def __init__(self):
        super().__init__()

    def _get_valid_setitem_dtypes(self):
        return [PcfTensor, Pcf, Pcf32Tensor, Pcf64Tensor]

    def _decay_value(self, val):
        return val._data

    def _represent_element(self, element):
        return Pcf(element)

class Pcf32Tensor(PcfTensor):
    def __init__(self, data : cpp.Pcf32Tensor):
        super().__init__()
        self._data = data
        self.dtype = pcf32

    def _to_py_tensor(self, data):
        return Pcf32Tensor(data)

class Pcf64Tensor(PcfTensor):
    def __init__(self, data : cpp.Pcf64Tensor):
        super().__init__()
        self._data = data
        self.dtype = pcf64

    def _to_py_tensor(self, data):
        return Pcf64Tensor(data)

class PointCloudTensor(Tensor):
    def _get_valid_setitem_dtypes(self):
        return [Float64Tensor, Float32Tensor, np.ndarray, float, int]

class PointCloud32Tensor(PointCloudTensor):
    def __init__(self, data : cpp.PointCloud32Tensor | PointCloud32Tensor):
        super().__init__()

        Tensor._validate_constructor_arg(self, data, [cpp.PointCloud32Tensor, PointCloud32Tensor])

        if isinstance(data, cpp.PointCloud32Tensor):
            self._data = data
        elif isinstance(data, PointCloud32Tensor):
            self._data = data._data
        else:
            raise TypeError(f'Internal type error, please report this: Unhandled {type(data)}')

        self.dtype = pcloud32

    def _to_py_tensor(self, data):
        return PointCloud32Tensor(data)

    def _represent_element(self, element):
        return Float32Tensor(element)

    def _decay_value(self, val):
        t = Float32Tensor(val)
        return t._data

class PointCloud64Tensor(PointCloudTensor):
    def __init__(self, data : cpp.PointCloud64Tensor):
        super().__init__()
        Tensor._validate_constructor_arg(self, data, [cpp.PointCloud64Tensor, PointCloud64Tensor])

        if isinstance(data, cpp.PointCloud64Tensor):
            self._data = data
        elif isinstance(data, PointCloud64Tensor):
            self._data = data._data
        else:
            raise TypeError(f'Internal type error, please report this: Unhandled {type(data)}')

        self.dtype = pcloud64

    def _to_py_tensor(self, data):
        return PointCloud64Tensor(data)

    def _represent_element(self, element):
        t = Float64Tensor(element)
        return Float64Tensor(element)

    def _decay_value(self, val):
        t = Float64Tensor(val)
        return t._data




PcfContainerLike = Union[Tensor, list[Pcf], Pcf]

def _to_tensor_pcf(fs : PcfContainerLike):
    if isinstance(fs, PcfTensor):
        return fs

    # TODO: Deal with lists/single pcfs

    raise TypeError('Input should be convertible to a PcfTensor.')

def _get_backend(fs, backendMapping : dict):
    if isinstance(fs, Tensor):
        _validate_dtype(fs.dtype, backendMapping.keys())

        backend = backendMapping.get(fs.dtype)
        if backend is None:
            raise ValueError(f'Operation not supported for tensors of this type ({fs._type} with dtype {fs.dtype})')
        return backend, fs
    elif isinstance(fs, np.ndarray):
        if fs.dtype == np.float32:
            return _get_backend(Float32Tensor(fs), backendMapping)
        elif fs.dtype == np.float64:
            return _get_backend(Float64Tensor(fs), backendMapping)
    elif hasattr(fs, 'dtype'):
        _validate_dtype(fs.dtype, backendMapping.keys())
        backend = backendMapping.get(fs.dtype)
        if backend is None:
            raise ValueError(f'Operation not supported for objects of this type ({type(fs)} with dtype {fs.dtype})')
        return backend, fs

    raise ValueError(f'Operation not supported for data of this type ({type(fs)})')
