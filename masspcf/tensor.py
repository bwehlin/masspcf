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

from . import mpcf_cpp as cpp
from .typing import float32, float64

from abc import ABC, abstractmethod
from typing import Union
import numpy as np

#class Tensor:
#    def __init__(self):
#        pass

TShape = cpp.TShape

TShapeLike = Union[TShape, tuple[int, ...]]

def _pyslice_to_slice(s):
    if isinstance(s, int):
        return cpp.slice_index(s)
    elif isinstance(s, slice):
        step = 1 if s.step is None else s.step

        if s.start is None and s.stop is None:
            return cpp.slice_all()
        elif s.start is not None and s.stop is not None:
            return cpp.slice_range(s.start, step, s.stop)
    
    raise TypeError("Unhandled slice type")

class Tensor(ABC):
    def __getitem__(self, slices):
        if all(isinstance(slice, int) for slice in slices):
            return self._data._get_element(slices)
        else:
            real_slices = [_pyslice_to_slice(slice) for slice in slices]
            return self._getitem(real_slices)

    def __setitem__(self, slices, val):
        if all(isinstance(slice, int) for slice in slices):
            self._data._set_element(slices, val)
        else:
            raise ValueError("Unimplemented.")

    @abstractmethod
    def _getitem(self, slices):
        pass

    @property
    def shape(self) -> TShape:
        return self._data.shape
    
    @property
    def strides(self):
        return self._data.strides

class NumericTensor(Tensor):
    pass

class FloatTensor(NumericTensor):
    def __init__(self, data : cpp.FloatTensor):
        self._data = data
        self.dtype = float32
    
    def _getitem(self, slices):
        return FloatTensor(self._data[slices])

class DoubleTensor(NumericTensor):
    def __init__(self, data : cpp.DoubleTensor):
        self._data = data
        self.dtype = float64
    
    def _getitem(self, slices):
        return DoubleTensor(self._data[slices])

def zerosT(shape : TShapeLike, dtype=float64):
    if not isinstance(shape, TShape):
        shape = TShape(shape) # If passed as, e.g., tuple of ints

    if dtype == float32:
        return FloatTensor(cpp.FloatTensor(shape, 0.0))
    elif dtype == float64:
        return DoubleTensor(cpp.DoubleTensor(shape, 0.0))
    else:
        raise TypeError("Only float32/float64 are supported for dtype.")
