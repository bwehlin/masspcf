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

from abc import ABC, abstractmethod
from typing import Union

from . import _mpcf_cpp as cpp

Shape = cpp.Shape

ShapeLike = Shape | tuple[int, ...]


def _unpickle_tensor(data: bytes):
    import io as _io
    from .io import _load
    return _load(_io.BytesIO(data))

CppTensor = Union[
    cpp.Float32Tensor,
    cpp.Float64Tensor,
    cpp.Int32Tensor,
    cpp.Int64Tensor,
    cpp.Uint32Tensor,
    cpp.Uint64Tensor,
    cpp.Pcf32Tensor,
    cpp.Pcf64Tensor,
    cpp.Pcf32iTensor,
    cpp.Pcf64iTensor,
    cpp.PointCloud32Tensor,
    cpp.PointCloud64Tensor,
    cpp.BoolTensor,
]


def _infer_shape_and_flatten(data):
    """Walk a nested list/tuple structure and return (shape, flat_elements).

    Recursion stops at any element that is not a list or tuple.
    Validates that the structure is rectangular.
    """
    shape: list[int] = []

    def _probe(obj, depth):
        if not isinstance(obj, (list, tuple)):
            return
        if depth == len(shape):
            shape.append(len(obj))
        elif shape[depth] != len(obj):
            raise ValueError(
                f"Ragged nested list: expected length {shape[depth]} at depth {depth}, got {len(obj)}"
            )
        if obj:
            _probe(obj[0], depth + 1)

    _probe(data, 0)

    flat: list = []

    def _collect(obj, depth):
        if depth == len(shape):
            flat.append(obj)
        else:
            for item in obj:
                _collect(item, depth + 1)

    _collect(data, 0)
    return tuple(shape), flat


def _tensor_from_nested(data, elem_to_tensor, default_ctor=None):
    """Build a C++ tensor from a nested list/tuple of elements with ``._data``.

    *elem_to_tensor* maps C++ element type to C++ tensor constructor,
    e.g. ``{cpp.Pcf_f32_f32: cpp.Pcf32Tensor}``.
    *default_ctor* is used when the list is empty (no element to infer from).
    """
    shape, flat = _infer_shape_and_flatten(data)
    if not flat:
        if default_ctor is None:
            default_ctor = next(iter(elem_to_tensor.values()))
        return default_ctor(cpp.Shape(list(shape or (0,))))
    tensor_ctor = elem_to_tensor.get(type(flat[0]._data))
    if tensor_ctor is None:
        raise TypeError(f"Unsupported element type {type(flat[0])}")
    t = tensor_ctor(cpp.Shape([len(flat)]))
    for i, elem in enumerate(flat):
        t._set_element([i], elem._data)
    if shape != (len(flat),):
        t = t.reshape(list(shape))
    return t


def _pyslice_to_slice(s):
    if isinstance(s, int):
        return cpp.slice_index(s)
    elif isinstance(s, slice):
        return cpp.slice_range(s.start, s.stop, s.step)

    raise TypeError("Unhandled slice type")


def _resolve_negative_indices(index_tensor, axis_size):
    """Resolve negative indices and bounds-check."""
    import numpy as np
    from .tensor import IntTensor
    arr = np.asarray(index_tensor).astype(np.int64).copy()
    neg = arr < 0
    arr[neg] += axis_size
    if np.any((arr < 0) | (arr >= axis_size)):
        raise IndexError(f"Index out of bounds for axis with size {axis_size}")
    return IntTensor(arr)


class Tensor(ABC):
    _data: CppTensor

    __array_ufunc__ = None

    def __array__(self, dtype=None, copy=None):
        raise TypeError(
            f"np.asarray() is not supported for {type(self).__name__}. "
            f"Only numeric tensors (FloatTensor, IntTensor, BoolTensor) "
            f"can be converted to NumPy arrays."
        )

    @staticmethod
    def _coerce_index_arrays(slices):
        """Convert numpy bool/int arrays in an index tuple to BoolTensor/IntTensor."""
        import numpy as np
        from .tensor import BoolTensor, IntTensor
        if not isinstance(slices, tuple):
            slices = (slices,)
        result = []
        for s in slices:
            if isinstance(s, np.ndarray):
                if s.dtype == np.bool_:
                    result.append(BoolTensor(s))
                elif np.issubdtype(s.dtype, np.integer):
                    result.append(IntTensor(s))
                else:
                    result.append(s)
            else:
                result.append(s)
        return tuple(result)

    def __getitem__(self, slices):
        from .tensor import BoolTensor, IntTensor

        slices = self._coerce_index_arrays(slices)

        # Collect advanced index positions (BoolTensor or IntTensor)
        advanced = [(i, s) for i, s in enumerate(slices) if isinstance(s, (BoolTensor, IntTensor))]

        if not advanced:
            return self._getitem_slices(slices)

        # Single full-shape BoolTensor: flat masked select
        if len(slices) == 1 and isinstance(slices[0], BoolTensor):
            return self._to_py_tensor(self._data.masked_select(slices[0]._data))  # type: ignore[arg-type]

        # Apply plain slices first, leaving advanced index axes as slice(None)
        slice_parts = tuple(slice(None) if isinstance(s, (BoolTensor, IntTensor)) else s for s in slices)
        result = self._getitem_slices(slice_parts)

        # Apply each advanced index sequentially (outer indexing)
        for orig_pos, idx in advanced:
            dims_dropped = sum(1 for j in range(orig_pos) if isinstance(slices[j], int))
            axis = orig_pos - dims_dropped
            if isinstance(idx, BoolTensor):
                result = self._to_py_tensor(result._data.axis_select(axis, idx._data))  # type: ignore[arg-type]
            else:
                resolved = _resolve_negative_indices(idx, result.shape[axis])
                result = self._to_py_tensor(result._data.index_select(axis, resolved._data))

        return result

    def _getitem_slices(self, slices):
        """Handle indexing with only int/slice components (no BoolTensor)."""
        if len(slices) == 1 and isinstance(slices[0], int):
            if len(self.shape) == 1:
                return self._represent_element(self._data._get_element(slices[0]))
            # Multi-dim: single int selects along axis 0 → return a sub-tensor
            idx = slices[0]
            s = _pyslice_to_slice(slice(idx, idx + 1 if idx != -1 else None))
            return self._to_py_tensor(self._data[[s]]).squeeze(0)
        if len(slices) == 1 and isinstance(slices[0], slice):
            return self._to_py_tensor(self._data[[_pyslice_to_slice(slices[0])]])
        if all(isinstance(s, int) for s in slices):
            return self._represent_element(self._data._get_element(slices))
        real_slices = [_pyslice_to_slice(s) for s in slices]
        return self._to_py_tensor(self._data[real_slices])

    @abstractmethod
    def _get_valid_setitem_dtypes(self):
        raise NotImplementedError()

    def _validate_setitem_dtype(self, val):
        valid_dtypes = self._get_valid_setitem_dtypes()
        if not any(isinstance(val, dt) for dt in valid_dtypes):
            raise TypeError(
                f"Tried to set an item of a tensor of type {type(self)} to a value of type {type(val)}. Only {valid_dtypes} are accepted."
            )

    def __setitem__(self, slices, val):
        from .tensor import BoolTensor, IntTensor

        slices = self._coerce_index_arrays(slices)

        advanced = [(i, s) for i, s in enumerate(slices) if isinstance(s, (BoolTensor, IntTensor))]

        self._validate_setitem_dtype(val)

        if not advanced:
            self._setitem_slices(slices, val)
            return

        # Single full-shape BoolTensor: flat masked assign/fill
        if len(slices) == 1 and isinstance(slices[0], BoolTensor):
            if isinstance(val, Tensor):
                self._data.masked_assign(slices[0]._data, val._data)  # type: ignore[arg-type]
            else:
                self._data.masked_fill(slices[0]._data, self._decay_value(val))  # type: ignore[arg-type]
            return

        # Apply plain slices first to get a mutable view
        slice_parts = tuple(slice(None) if isinstance(s, (BoolTensor, IntTensor)) else s for s in slices)
        view = self._getitem_slices(slice_parts)

        # Build selectors for outer indexing
        selectors = []
        for orig_pos, idx in advanced:
            dims_dropped = sum(1 for j in range(orig_pos) if isinstance(slices[j], int))
            axis = orig_pos - dims_dropped
            if isinstance(idx, BoolTensor):
                selectors.append((axis, idx._data))
            else:
                resolved = _resolve_negative_indices(idx, view.shape[axis])
                selectors.append((axis, resolved._data))

        if len(selectors) == 1:
            axis, sel_data = selectors[0]
            if isinstance(slices[advanced[0][0]], BoolTensor):
                if isinstance(val, Tensor):
                    view._data.axis_assign(axis, sel_data, val._data)  # type: ignore[arg-type]
                else:
                    view._data.axis_fill(axis, sel_data, self._decay_value(val))  # type: ignore[arg-type]
            else:
                if isinstance(val, Tensor):
                    view._data.index_assign(axis, sel_data, val._data)  # type: ignore[arg-type]
                else:
                    view._data.index_fill(axis, sel_data, self._decay_value(val))  # type: ignore[arg-type]
        else:
            if isinstance(val, Tensor):
                view._data.outer_assign(selectors, val._data)  # type: ignore[arg-type]
            else:
                view._data.outer_fill(selectors, self._decay_value(val))  # type: ignore[arg-type]

    def _setitem_slices(self, slices, val):
        """Handle setitem with only int/slice components (no BoolTensor)."""
        if len(slices) == 1 and isinstance(slices[0], int):
            self._data._set_element([slices[0]], self._decay_value(val))
        elif len(slices) == 1 and isinstance(slices[0], slice):
            real_slices = [_pyslice_to_slice(slices[0])]
            self._data[real_slices] = self._decay_value(val)
        elif all(isinstance(s, int) for s in slices):
            self._data._set_element(slices, self._decay_value(val))
        else:
            real_slices = [_pyslice_to_slice(s) for s in slices]
            self._data[real_slices] = self._decay_value(val)

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def __eq__(self, rhs):
        if not isinstance(rhs, Tensor):
            return NotImplemented
        from .tensor import BoolTensor
        return BoolTensor(self._data == rhs._data)  # type: ignore[arg-type]

    def __ne__(self, rhs):
        if not isinstance(rhs, Tensor):
            return NotImplemented
        from .tensor import BoolTensor
        return BoolTensor(self._data != rhs._data)  # type: ignore[arg-type]

    def __lt__(self, rhs):
        from .tensor import BoolTensor
        return BoolTensor(self._data < rhs._data)

    def __le__(self, rhs):
        from .tensor import BoolTensor
        return BoolTensor(self._data <= rhs._data)

    def __gt__(self, rhs):
        from .tensor import BoolTensor
        return BoolTensor(self._data > rhs._data)

    def __ge__(self, rhs):
        from .tensor import BoolTensor
        return BoolTensor(self._data >= rhs._data)

    def array_equal(self, rhs) -> bool:
        """Test whether two tensors have the same shape and all equal elements.

        Parameters
        ----------
        rhs : Tensor
            The tensor to compare with.

        Returns
        -------
        bool
            True if the tensors are elementwise equal, False otherwise.
        """
        return self._data.array_equal(rhs._data)

    def __reduce__(self):
        import io as _io
        from .io import _save, _load
        buf = _io.BytesIO()
        _save(self, buf)
        return _unpickle_tensor, (buf.getvalue(),)

    def __deepcopy__(self, memodict=None):
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

    def broadcast_to(self, shape):
        """Return a broadcast view of this tensor with the given target shape.

        Dimensions of size 1 are expanded (stride set to 0); prepended
        dimensions also get stride 0. No data is copied — the result shares
        the underlying storage.

        Parameters
        ----------
        shape : tuple of int
            The target shape. Must be broadcast-compatible with the current
            shape.

        Returns
        -------
        Tensor
            A non-contiguous view of this tensor with the target shape.

        Raises
        ------
        ValueError
            If the shapes are not broadcast-compatible.
        """
        return self._to_py_tensor(self._data.broadcast_to(list(shape)))

    def flatten(self):
        return self._to_py_tensor(self._data.flatten())

    def reshape(self, shape):
        return self._to_py_tensor(self._data.reshape(list(shape)))

    def transpose(self, axes=None):
        return self._to_py_tensor(self._data.transpose(list(axes) if axes else []))

    @property
    def T(self):
        return self.transpose()

    def swapaxes(self, axis1, axis2):
        n = self.ndim
        if axis1 < 0:
            axis1 += n
        if axis2 < 0:
            axis2 += n
        if not (0 <= axis1 < n and 0 <= axis2 < n):
            raise IndexError(
                f"swapaxes: axis out of range for tensor with {n} dimensions")
        return self._to_py_tensor(self._data.swapaxes(axis1, axis2))

    def squeeze(self, axis=None):
        if axis is None:
            return self._to_py_tensor(self._data.squeeze())
        return self._to_py_tensor(self._data.squeeze(axis))

    def expand_dims(self, axis):
        return self._to_py_tensor(self._data.expand_dims(axis))

    # Implementation: looks up a C++ cast function by naming convention:
    # cpp.cast_{src_tag}_{dst_tag}. To add a new cast:
    #   1. Ensure the C++ target type is constructible from the source
    #      (add a converting constructor if needed).
    #   2. Add a tensor_cast binding in py_tensor.cpp named cast_{src}_{dst}.
    #   3. Add entries to _DTYPE_TAG and _DTYPE_TO_WRAPPER in typing.py
    #      if the dtype is new.
    def astype(self, dtype):
        """Return a new tensor with elements cast to the given dtype.

        Supported casts are same-family precision changes (e.g. float32 → float64,
        pcf32 → pcf64) and numeric cross-family (e.g. float → int).
        """
        if dtype == self.dtype:
            return self.copy()
        from . import _mpcf_cpp as cpp
        from .typing import _DTYPE_TAG, _DTYPE_TO_WRAPPER, _init_dtype_wrappers
        _init_dtype_wrappers()
        src_tag = _DTYPE_TAG.get(self.dtype)
        dst_tag = _DTYPE_TAG.get(dtype)
        if src_tag is None or dst_tag is None:
            raise TypeError(
                f"Cannot cast from {self.dtype.__name__} to {dtype.__name__}")
        cast_fn = getattr(cpp, f"cast_{src_tag}_{dst_tag}", None)
        if cast_fn is None:
            raise TypeError(
                f"Cannot cast from {self.dtype.__name__} to {dtype.__name__}")
        return _DTYPE_TO_WRAPPER[dtype](cast_fn(self._data))

    @property
    def shape(self) -> Shape:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return len(self._data.shape)

    @property
    def size(self) -> int:
        s = self._data.shape
        result = 1
        for i in range(len(s)):
            result *= s[i]
        return result

    def __len__(self) -> int:
        return self._data.shape[0]

    @property
    def strides(self):
        return self._data.strides

    @property
    def offset(self):
        return self._data.offset

    @staticmethod
    def _validate_constructor_arg(tensor, arg, valid_types):
        if not any(type(arg) is tp for tp in valid_types):
            raise TypeError(
                f"Tried to construct tensor of type {type(tensor)} from argument of type {type(arg)}. Only the following type(s) are allowed: {valid_types}."
            )


class FunctionTensorMixin:
    """Mixin for tensors whose elements can be evaluated at domain points.

    Delegates to C++ ``__call__`` overloads which handle scalar, numpy array,
    and Tensor inputs. Tensor inputs are unwrapped, evaluated in C++,
    and the result is re-wrapped via the input's ``_to_py_tensor``.
    For scalars, lists, and numpy arrays the result is returned as an ndarray.
    """

    def __call__(self, t):
        """Evaluate every element of the tensor at the given domain point(s).

        Parameters
        ----------
        t : scalar, list, numpy.ndarray, or NumericTensor
            A single domain value or a collection of values.

        Returns
        -------
        numpy.ndarray or NumericTensor
            For scalar *t*, the result has shape ``self.shape``.
            For array-like *t* of shape ``t_shape``, the result has shape
            ``self.shape + t_shape``.
            A ``NumericTensor`` input produces a ``NumericTensor`` output;
            all other inputs produce a ``numpy.ndarray``.
        """
        import numpy as np

        if isinstance(t, Tensor):
            return t._to_py_tensor(self._data(t._data))
        if isinstance(t, list):
            t = np.asarray(t)
        if isinstance(t, (int, float, np.generic, np.ndarray)):
            return np.asarray(self._data(t))
        raise TypeError(f"Cannot evaluate tensor at argument of type {type(t)}")


class ArithmeticTensorMixin:
    """Mixin providing elementwise arithmetic operators for tensors.

    Operators accept either a scalar or another tensor of the same type.
    When both operands are tensors, NumPy-style broadcasting is applied:
    shapes are compared right-to-left, dimensions match when equal or one
    is 1, and missing leading dimensions are treated as size 1.

    In-place operators (``+=``, ``-=``, ``*=``, ``/=``) require that the
    broadcast output shape equals the shape of the left-hand operand
    (the left-hand side is never expanded).
    """

    def _decay_operand(self, val):
        return val._data if hasattr(val, "_data") else val

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

    def __neg__(self):
        return self._to_py_tensor(-self._data)

    def __truediv__(self, rhs):
        return self._to_py_tensor(self._data / self._decay_operand(rhs))

    def __rtruediv__(self, lhs):
        return self._to_py_tensor(self._decay_operand(lhs) / self._data)

    def __itruediv__(self, rhs):
        self._data /= self._decay_operand(rhs)
        return self

    def __pow__(self, exponent):
        """Raise every element of the tensor to a power.

        Returns a new tensor whose elements are each raised to
        ``exponent``. A ``RuntimeWarning`` is emitted if the result
        contains NaN or infinity.

        Parameters
        ----------
        exponent : float or int
            The exponent.

        Returns
        -------
        Tensor
            A new tensor with transformed elements.
        """
        return self._to_py_tensor(self._data.__pow__(exponent))

    def __ipow__(self, exponent):
        """Raise every element of the tensor to a power in place.

        Parameters
        ----------
        exponent : float or int
            The exponent.

        Returns
        -------
        self
        """
        self._data.__ipow__(exponent)
        return self
