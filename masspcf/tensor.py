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

from typing import Union
import numpy as np

#class Tensor:
#    def __init__(self):
#        pass

TShape = cpp.TShape

TShapeLike = Union[TShape, tuple[int, ...]]

class NumericTensor:
    @property
    def shape(self) -> TShape:
        return self.data_.shape

class FloatTensor(NumericTensor):
    def __init__(self, shape : TShape, init : float = 0.0):
        self.data_ = cpp.FloatTensor(shape, init)
        self.dtype = float32

class DoubleTensor(NumericTensor):
    def __init__(self, shape : TShape, init : float = 0.0):
        self.data_ = cpp.DoubleTensor(shape, init)
        self.dtype = float64

def zerosT(shape : TShapeLike, dtype=float64):
    if not isinstance(shape, TShape):
        shape = TShape(shape) # If passed as, e.g., tuple of ints

    if dtype == float32:
        return FloatTensor(shape)
    elif dtype == float64:
        return DoubleTensor(shape)
    else:
        raise TypeError("Only float32/float64 are supported for dtype.")
