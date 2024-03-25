'''
    Copyright 2024 Bjorn Wehlin

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
'''

from . import mpcf_cpp as cpp
from .pcf import Pcf, tPcf_f32_f32, tPcf_f64_f64
import numpy as np
#from numpy.typing import _ShapeLike

from . import typing as dt

class Shape:
    def __init__(self, s):
        if isinstance(s, Shape):
            self.data = s.data.copy()
        elif isinstance(s, cpp.Shape):
            self.data = s
        else:
            # TODO: check that s is a tuple of unsigned ints
            self.data = cpp.Shape(list(s))

    def __str__(self):
        return 'Shape(' + ', '.join([ str(self.data.at(i)) for i in range(self.data.size()) ]) + ')'

def _get_underlying_shape(s):
    if isinstance(s, Shape):
        return s.data
    elif isinstance(s, cpp.Shape):
        return s
    else:
        return Shape(s).data

class Array:
    def __init__(self, data):
        self.data = data
    
    def shape(self):
        return Shape(self.data.shape())

def _get_array_class(dtype):
    if dtype == dt.float32:
        return cpp.NdArray_f32_f32
    elif dtype == dt.float64:
        return cpp.NdArray_f64_f64
    else:
        raise TypeError('Only float32 and float64 dtypes are supported.')

def zeros(shape, dtype=dt.float32):
    ac = _get_array_class(dtype)
    return Array(ac.make_zeros(_get_underlying_shape(shape)))


#def zeros(size : _ShapeLike, dtype=):
#    return Array(np.zeros(size, dtype=))