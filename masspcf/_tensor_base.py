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

from __future__ import annotations

from . import _mpcf_cpp as cpp

from abc import ABC, abstractmethod
from typing import Union

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
            x = self._data._get_element(slices)
            return self._represent_element(x)
        elif isinstance(slices, slice): # X[n:m] etc...
            x = self._data[[_pyslice_to_slice(slices)]]
            return self._to_py_tensor(x)
        elif all(isinstance(s, int) for s in slices): # X[1, 2, 3] etc... (for this, we wan't a single element rather than a tensor)
            x = self._data._get_element(slices)
            return self._represent_element(x)
        else:
            real_slices = [_pyslice_to_slice(s) for s in slices]
            return self._to_py_tensor(self._data[real_slices])

    @abstractmethod
    def _get_valid_setitem_dtypes(self):
        raise NotImplementedError()

    def _validate_setitem_dtype(self, val):
        valid_dtypes = self._get_valid_setitem_dtypes()
        if not any(isinstance(val, dt) for dt in valid_dtypes):
            raise TypeError(f'Tried to set an item of a tensor of type {type(self)} to a value of type {type(val)}. Only {valid_dtypes} are accepted.')

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
        """
        Convert C++/Python tensor type to a Python tensor. Normally, it is enough to just return Datatype(data) where
        Datatype is the current class's type.
        """
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
        """
        Casts a single C++ element into the corresponding Python type. For example, a C++ function may return a
        mpcf._mpcf_cpp.Pcf32, and then we would like to cast it into a mpcf.Pcf by simply returning mpcf.Pcf(element).
        """
        raise NotImplementedError()

    def _decay_operand(self, val):
        return val._data if hasattr(val, '_data') else val

    def __add__(self, rhs):
        return self._to_py_tensor(self._data + self._decay_operand(rhs))

    def __radd__(self, lhs):
        return self._to_py_tensor(self._decay_operand(lhs) + self._data)

    def __iadd__(self, rhs):
        self._data += self._decay_operand(rhs)
        return self

    def __sub__(self, rhs):
        return self._to_py_tensor(self._data - self._decay_operand(rhs))

    def __rsub__(self, lhs):
        return self._to_py_tensor(self._decay_operand(lhs) - self._data)

    def __isub__(self, rhs):
        self._data -= self._decay_operand(rhs)
        return self

    def __mul__(self, rhs):
        return self._to_py_tensor(self._data * self._decay_operand(rhs))

    def __rmul__(self, lhs):
        return self._to_py_tensor(self._decay_operand(lhs) * self._data)

    def __imul__(self, rhs):
        self._data *= self._decay_operand(rhs)
        return self

    def __truediv__(self, rhs):
        return self._to_py_tensor(self._data / self._decay_operand(rhs))

    def __itruediv__(self, rhs):
        self._data /= self._decay_operand(rhs)
        return self

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

    @staticmethod
    def _validate_constructor_arg(tensor, arg, valid_types):
        if not any(type(arg) == tp for tp in valid_types):
            raise TypeError(f'Tried to construct tensor of type {type(tensor)} from argument of type {type(arg)}. Only the following type(s) are allowed: {valid_types}.')

