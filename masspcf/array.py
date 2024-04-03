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

    def __len__(self):
        return self.data.size()

    def __getitem__(self, i):
        return self.data.at(i)

class Container:
    @property
    def shape(self):
        return Shape(self._get_data().shape())
    
    def _get_slice_vec(self, pos):
        sv = cpp.StridedSliceVector()
        
        if isinstance(pos, slice):
            pos = (pos,)

        for p in pos:
            if isinstance(p, int):
                sv.append(p)
            elif isinstance(p, slice):
                if p.start is None and p.stop is None:
                    sv.append_all()
            elif p == Ellipsis:
                sv.append_all()
            else:
                raise ValueError(f'Unsupported range construct {p}.')

        return sv
    
    def at(self, pos):
        return self.data.at(pos)

    def __getitem__(self, pos):
        if isinstance(pos, int):
            return self._as_view().at([pos])

        sv = self._get_slice_vec(pos)
        view = View(self.data.strided_view(sv))
        
        if len(view.shape) == 0:
            return view.at([0])
        else:
            return view
    
    def __setitem__(self, pos, val):
        sv = self._get_slice_vec(pos)
        self._get_data().strided_view(sv).assign(val._as_view().data)
    
    def _get_data(self):
        return self._as_view().data
        

class View(Container):
    def __init__(self, v):
        self.data = v
    
    def _as_view(self):
        return self

def _get_underlying_shape(s):
    if isinstance(s, Shape):
        return s.data
    elif isinstance(s, cpp.Shape):
        return s
    else:
        return Shape(s).data

class Array(Container):
    def __init__(self, data):
        self.data = data
    
    def _as_view(self):
        return View(self.data.as_view())

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

def mean(A, dim):
    return Array(A._as_view().data.reduce_mean(dim))
