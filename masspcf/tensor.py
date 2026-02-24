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

from . import _mpcf_cpp as cpp
from .typing import pcf32, pcf64, f32, f64, _check_deprecated_dtype, _assert_valid_dtype, _validate_dtype
from .pcf import Pcf

from abc import ABC, abstractmethod
from typing import Union
import numpy as np

from enum import Enum

Shape = cpp.Shape

ShapeLike = Union[Shape, tuple[int, ...]]

def _pyslice_to_slice(s):
    if isinstance(s, int):
        return cpp.slice_index(s)
    elif isinstance(s, slice):
        return cpp.slice_range(s.start, s.stop, s.step)
    
    raise TypeError("Unhandled slice type")

class Tensor(ABC):
    @abstractmethod
    def _to_py_tensor(self, data):
        raise NotImplementedError()

    def __getitem__(self, slices):
        if isinstance(slices, int): # X[n]
            return self._represent_element(self._data._get_element(slices))
        elif isinstance(slices, slice): # X[n:m] etc...
            return self._getitem([_pyslice_to_slice(slices)])
        elif all(isinstance(s, int) for s in slices): # X[1, 2, 3] etc... (for this, we wan't a single element rather than a tensor)
            return self._represent_element(self._data._get_element(slices))
        else:
            real_slices = [_pyslice_to_slice(s) for s in slices]
            return self._getitem(real_slices)

    @abstractmethod
    def _get_valid_setitem_dtypes(self):
        raise NotImplementedError()

    def _validate_setitem_dtype(self, val):
        valid_dtypes = self._get_valid_setitem_dtypes()
        if not any(isinstance(val, dt) for dt in valid_dtypes):
            raise TypeError(f'Tried to assign value of type {type(val)} to an object of type {type(self)}. Only {valid_dtypes} are accepted.')

    def __setitem__(self, slices, val):
        self._validate_setitem_dtype(val)

        if isinstance(slices, int):
            self._data._set_element([slices], self._decay_value(val))
        elif all(isinstance(s, int) for s in slices):
            self._data._set_element(slices, self._decay_value(val))
        # TODO: single slice
        else:
            real_slices = [_pyslice_to_slice(s) for s in slices]
            self._data[real_slices] = val._data # TODO: 32/64 conversion, to_tensor, etc.

    def __eq__(self, rhs):
        return self._data == rhs._data

    def __deepcopy__(self, memodict={}):
        return self._to_py_tensor(self._data.copy())

    def copy(self):
        return self.__deepcopy__()

    @abstractmethod
    def _decay_value(self, val):
        raise NotImplementedError()

    @abstractmethod
    def _getitem(self, slices):
        raise NotImplementedError()

    @abstractmethod
    def _represent_element(self, element):
        raise NotImplementedError()

    @abstractmethod
    def flatten(self):
        raise NotImplementedError()

    @property
    def shape(self) -> Shape:
        return self._data.shape
    
    @property
    def strides(self):
        return self._data.strides
    
    @property
    def offset(self):
        return self._data.offset


class NumericTensor(Tensor):
    def __init__(self):
        super().__init__()

    def _get_valid_setitem_dtypes(self):
        return [NumericTensor, FloatTensor, DoubleTensor, float, int, np.ndarray]

    def _decay_value(self, val):
        return val

    def _represent_element(self, element):
        return element

    def __array__(self):
        return np.array(self._data)

class FloatTensor(NumericTensor):
    def __init__(self, data : cpp.FloatTensor):
        super().__init__()
        self._data = data
        self.dtype = f32

    def _to_py_tensor(self, data):
        return FloatTensor(data)
    
    def _getitem(self, slices):
        return FloatTensor(self._data[slices])
    
    def flatten(self):
        return FloatTensor(self._data.flatten())

class DoubleTensor(NumericTensor):
    def __init__(self, data : cpp.DoubleTensor):
        super().__init__()
        self._data = data
        self.dtype = f64

    def _to_py_tensor(self, data):
        return DoubleTensor(data)

    def _getitem(self, slices):
        return DoubleTensor(self._data[slices])

    def flatten(self):
        return DoubleTensor(self._data.flatten())

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

    def _getitem(self, slices):
        return Pcf32Tensor(self._data[slices])

    def flatten(self):
        return Pcf32Tensor(self._data.flatten())

class Pcf64Tensor(PcfTensor):
    def __init__(self, data : cpp.Pcf64Tensor):
        super().__init__()
        self._data = data
        self.dtype = pcf64

    def _to_py_tensor(self, data):
        return Pcf64Tensor(data)

    def _getitem(self, slices):
        return Pcf64Tensor(self._data[slices])

    def flatten(self):
        return Pcf64Tensor(self._data.flatten())

def zeros(shape : ShapeLike, dtype=pcf32):
    if not isinstance(shape, Shape):
        shape = Shape(shape) # If passed as, e.g., tuple of ints

    _check_deprecated_dtype(dtype)
    _assert_valid_dtype(dtype, [pcf32, pcf64, f32, f64])

    if dtype == pcf32:
        return Pcf32Tensor(cpp.Pcf32Tensor(shape))
    elif dtype == pcf64:
        return Pcf64Tensor(cpp.Pcf64Tensor(shape))
    elif dtype == f32:
        return FloatTensor(cpp.DoubleTensor(shape, 0.0))
    elif dtype == f64:
        return DoubleTensor(cpp.DoubleTensor(shape, 0.0))


PcfContainerLike = Union[Tensor, list[Pcf], Pcf]

def _to_tensor_pcf(fs : PcfContainerLike):
    if isinstance(fs, PcfTensor):
        return fs

    # TODO: Deal with lists/single pcfs

    raise TypeError('Input should be convertible to a PcfTensor.')

def _get_backend(fs : PcfContainerLike, backendMapping : dict):
    if isinstance(fs, Tensor):
        _validate_dtype(fs.dtype, backendMapping.keys())

        backend = backendMapping.get(fs.dtype)
        if backend is None:
            raise ValueError(f'Operation not supported for tensors of this type ({fs._type} with dtype {fs.dtype})')
        return backend

    raise ValueError(f'Operation not supported for data of this type ({type(fs)})')
