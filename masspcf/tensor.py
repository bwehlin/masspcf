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

from .typing import (pcf32, pcf64, f32, f64, pcloud32, pcloud64,
    float32, float64, # Deprecated types
    _check_deprecated_dtype, _assert_valid_dtype, _validate_dtype)

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
    def __getitem__(self, slices):
        if isinstance(slices, int): # X[n]
            return self._represent_element(self._data._get_element(slices))
        elif isinstance(slices, slice): # X[n:m] etc...
            return self._to_py_tensor(self._data[[_pyslice_to_slice(slices)]])
        elif all(isinstance(s, int) for s in slices): # X[1, 2, 3] etc... (for this, we wan't a single element rather than a tensor)
            return self._represent_element(self._data._get_element(slices))
        else:
            real_slices = [_pyslice_to_slice(s) for s in slices]
            return self._to_py_tensor(self._data[real_slices])

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
    def _to_py_tensor(self, data):
        raise NotImplementedError()

    @abstractmethod
    def _decay_value(self, val):
        """
        Convert a Python value into one that can be used by the corresponding C++ class. For example, if `X` is a Python
        `Tensor`, `_decay_value` should convert a Python value `val` so that the following (pseudocode) works:

        `X[1,2,3] = val`

        `X._data._set_element("1,2,3", self._decay_value(val))
        """
        raise NotImplementedError()

    @abstractmethod
    def _represent_element(self, element):
        raise NotImplementedError()

    def flatten(self):
        return self._to_py_tensor(self._data.flatten())

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
    def __init__(self, data : cpp.FloatTensor | FloatTensor | np.ndarray):
        super().__init__()

        if isinstance(data, cpp.FloatTensor):
            pass
        elif isinstance(data, FloatTensor):
            data = data._data
        elif isinstance(data, np.ndarray):
            data = cpp.ndarray_to_tensor_32(data)
        else:
            raise TypeError(f'Cannot create {type(self)} from {type(data)}')

        self._data = data
        self.dtype = f32

    def _to_py_tensor(self, data):
        return FloatTensor(data)

class DoubleTensor(NumericTensor):
    def __init__(self, data : cpp.DoubleTensor | DoubleTensor | np.ndarray):
        super().__init__()

        if isinstance(data, cpp.DoubleTensor):
            pass
        elif isinstance(data, DoubleTensor):
            data = data._data
        elif isinstance(data, np.ndarray):
            data = cpp.ndarray_to_tensor_64(data)
        else:
            raise TypeError(f'Cannot create {type(self)} from {type(data)}')

        self._data = data
        self.dtype = f64

    def _to_py_tensor(self, data):
        return DoubleTensor(data)

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
        return [np.ndarray, float, int]

class PointCloud32Tensor(PointCloudTensor):
    def __init__(self, data : cpp.PointCloud32Tensor):
        super().__init__()
        self._data = data
        self.dtype = pcloud32

    def _to_py_tensor(self, data):
        return PointCloud32Tensor(data)

    def _represent_element(self, element):
        return FloatTensor(element)

    def _decay_value(self, val):
        t = FloatTensor(val)
        return t._data

class PointCloud64Tensor(PointCloudTensor):
    def __init__(self, data : cpp.PointCloud64Tensor):
        super().__init__()
        self._data = data
        self.dtype = pcloud32

    def _to_py_tensor(self, data):
        return PointCloud64Tensor(data)

    def _represent_element(self, element):
        print(f'_represent_element {element} / {type(element)} -> {type(DoubleTensor(element))}')
        t = DoubleTensor(element)
        print(f'TP {type(t.shape)}')
        return DoubleTensor(element)

    def _decay_value(self, val):
        t = DoubleTensor(val)
        return t._data
        print(f'Decayed into {type(t._data)}')
        print(f'Decayed into {type(DoubleTensor(t._data)._data)}')
        return DoubleTensor(t._data)._data


def zeros(shape : ShapeLike, dtype=pcf32):
    """
    Creates a new `Tensor` of the specified `shape` and `dtype` whose entries are "zero." What "zero" means depends on the `dtype`:

    `dtype=pcf32/64`: A PCF that takes the value 0 for all times.
    `dtype=f32/f64`: The number 0
    `dtype=pcloud32/64`: An empty point cloud

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
    if not isinstance(shape, Shape):
        shape = Shape(shape) # If passed as, e.g., tuple of ints

    _check_deprecated_dtype(dtype)
    _assert_valid_dtype(dtype, [pcf32, pcf64, f32, f64, pcloud32, pcloud64, float32, float64])

    if dtype == pcf32 or dtype == float32:
        return Pcf32Tensor(cpp.Pcf32Tensor(shape))
    elif dtype == pcf64 or dtype == float64:
        return Pcf64Tensor(cpp.Pcf64Tensor(shape))
    elif dtype == f32:
        return FloatTensor(cpp.FloatTensor(shape, 0.0))
    elif dtype == f64:
        return DoubleTensor(cpp.DoubleTensor(shape, 0.0))
    elif dtype == pcloud32:
        return PointCloud32Tensor(cpp.PointCloud32Tensor(shape))
    elif dtype == pcloud64:
        return PointCloud64Tensor(cpp.PointCloud64Tensor(shape))
    else:
        raise NotImplementedError('This dtype has not been implemented.')


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
