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
from .typing import float32, float64
from .pcf import Pcf

from abc import ABC, abstractmethod
from typing import Union
import numpy as np

from enum import Enum

#class Tensor:
#    def __init__(self):
#        pass

TShape = cpp.TShape

TShapeLike = Union[TShape, tuple[int, ...]]

class TensorType(Enum):
    PCF = 1
    NUMERIC = 2

def _pyslice_to_slice(s):
    if isinstance(s, int):
        return cpp.slice_index(s)
    elif isinstance(s, slice):
        return cpp.slice_range(s.start, s.stop, s.step)
    
    raise TypeError("Unhandled slice type")

class Tensor(ABC):
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

    def __setitem__(self, slices, val):
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
    def shape(self) -> TShape:
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
        self._type = TensorType.NUMERIC
    
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
        self.dtype = float32
    
    def _getitem(self, slices):
        return FloatTensor(self._data[slices])
    
    def flatten(self):
        return FloatTensor(self._data.flatten())

class DoubleTensor(NumericTensor):
    def __init__(self, data : cpp.DoubleTensor):
        super().__init__()
        self._data = data
        self.dtype = float64
    
    def _getitem(self, slices):
        return DoubleTensor(self._data[slices])

    def flatten(self):
        return DoubleTensor(self._data.flatten())

class PcfTensor(Tensor):
    def __init__(self):
        super().__init__()
        self._type = TensorType.PCF
    
    def _decay_value(self, val):
        return val._data

    def _represent_element(self, element):
        return Pcf(element)

class Pcf32Tensor(PcfTensor):
    def __init__(self, data : cpp.Pcf32Tensor):
        super().__init__()
        self._data = data
        self.dtype = float32
    
    def _getitem(self, slices):
        return Pcf32Tensor(self._data[slices])

    def flatten(self):
        return Pcf32Tensor(self._data.flatten())

class Pcf64Tensor(PcfTensor):
    def __init__(self, data : cpp.Pcf64Tensor):
        super().__init__()
        self._data = data
        self.dtype = float64
    
    def _getitem(self, slices):
        return Pcf64Tensor(self._data[slices])

    def flatten(self):
        return Pcf64Tensor(self._data.flatten())

def zeros(shape : TShapeLike, dtype=float32, type=TensorType.PCF):
    if not isinstance(shape, TShape):
        shape = TShape(shape) # If passed as, e.g., tuple of ints

    if dtype == float32:
        match type:
            case TensorType.PCF:
                return Pcf32Tensor(cpp.Pcf32Tensor(shape))
            case TensorType.NUMERIC:
                return FloatTensor(cpp.FloatTensor(shape, 0.0))
            case _:
                raise ValueError("Unknown type for this dtype")
            
    elif dtype == float64:
        match type:
            case TensorType.PCF:
                return Pcf64Tensor(cpp.Pcf64Tensor(shape))
            case TensorType.NUMERIC:
                return FloatTensor(cpp.DoubleTensor(shape, 0.0))
            case _:
                raise ValueError("Unknown type for this dtype")
    else:
        raise TypeError("Only float32/float64 are supported for dtype.")

def zerosT(shape : TShapeLike, dtype=float64):
    if not isinstance(shape, TShape):
        shape = TShape(shape) # If passed as, e.g., tuple of ints

    if dtype == float32:
        return FloatTensor(cpp.FloatTensor(shape, 0.0))
    elif dtype == float64:
        return DoubleTensor(cpp.DoubleTensor(shape, 0.0))
    else:
        raise TypeError("Only float32/float64 are supported for dtype.")

PcfContainerLike = Union[Tensor, list[Pcf], Pcf]

def _to_tensor_pcf(fs : PcfContainerLike):
    if isinstance(fs, PcfTensor):
        return fs

    # TODO: Deal with lists/single pcfs

    raise TypeError('Input should be convertible to a PcfTensor.')

def _get_backend(fs : PcfContainerLike, backendMapping : dict):
    if isinstance(fs, Tensor):
        backend = backendMapping.get(( fs._type, fs.dtype ))
        if backend is None:
            raise ValueError(f'Operation not supported for tensors of this type ({fs._type} with dtype {fs.dtype})')
        return backend

    raise ValueError(f'Operation not supported for data of this type ({type(fs)})')
