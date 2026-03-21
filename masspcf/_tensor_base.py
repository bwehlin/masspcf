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

        # Find BoolTensor and IntTensor positions
        bool_positions = [(i, s) for i, s in enumerate(slices) if isinstance(s, BoolTensor)]
        int_positions = [(i, s) for i, s in enumerate(slices) if isinstance(s, IntTensor)]

        if bool_positions and int_positions:
            raise IndexError("Cannot mix BoolTensor and IntTensor indices")

        if not bool_positions and not int_positions:
            return self._getitem_slices(slices)

        # --- BoolTensor indexing ---
        if bool_positions:
            # Single full-shape BoolTensor: flat masked select
            if len(slices) == 1 and isinstance(slices[0], BoolTensor):
                return self._to_py_tensor(self._data.masked_select(slices[0]._data))  # type: ignore[arg-type]

            # Apply slices first, then apply bool masks (outer indexing)
            slice_parts = tuple(slice(None) if isinstance(s, BoolTensor) else s for s in slices)
            result = self._getitem_slices(slice_parts)

            if len(bool_positions) == 1:
                orig_pos, bool_tensor = bool_positions[0]
                dims_dropped = sum(1 for j in range(orig_pos) if isinstance(slices[j], int))
                axis = orig_pos - dims_dropped
                return self._to_py_tensor(result._data.axis_select(axis, bool_tensor._data))  # type: ignore[arg-type]

            axis_masks = []
            for orig_pos, bool_tensor in bool_positions:
                dims_dropped = sum(1 for j in range(orig_pos) if isinstance(slices[j], int))
                axis_masks.append((orig_pos - dims_dropped, bool_tensor._data))
            return self._to_py_tensor(result._data.multi_axis_select(axis_masks))

        # --- IntTensor advanced indexing ---
        if len(int_positions) > 1:
            raise IndexError("Only one IntTensor index per expression is supported")

        slice_parts = tuple(slice(None) if isinstance(s, IntTensor) else s for s in slices)
        result = self._getitem_slices(slice_parts)

        orig_pos, index_tensor = int_positions[0]
        dims_dropped = sum(1 for j in range(orig_pos) if isinstance(slices[j], int))
        axis = orig_pos - dims_dropped

        resolved = _resolve_negative_indices(index_tensor, result.shape[axis])
        return self._to_py_tensor(result._data.index_select(axis, resolved._data))

    def _getitem_slices(self, slices):
        """Handle indexing with only int/slice components (no BoolTensor)."""
        if len(slices) == 1 and isinstance(slices[0], int):
            return self._represent_element(self._data._get_element(slices[0]))
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

        # Find BoolTensor and IntTensor positions
        bool_positions = [(i, s) for i, s in enumerate(slices) if isinstance(s, BoolTensor)]
        int_positions = [(i, s) for i, s in enumerate(slices) if isinstance(s, IntTensor)]

        if bool_positions and int_positions:
            raise IndexError("Cannot mix BoolTensor and IntTensor indices")

        self._validate_setitem_dtype(val)

        if not bool_positions and not int_positions:
            self._setitem_slices(slices, val)
            return

        # --- BoolTensor indexing ---
        if bool_positions:
            # Single full-shape BoolTensor: flat masked assign/fill
            if len(slices) == 1 and isinstance(slices[0], BoolTensor):
                if isinstance(val, Tensor):
                    self._data.masked_assign(slices[0]._data, val._data)  # type: ignore[arg-type]
                else:
                    self._data.masked_fill(slices[0]._data, self._decay_value(val))  # type: ignore[arg-type]
                return

            # Mixed indexing: apply slices first to get a mutable view, then axis ops
            slice_parts = tuple(slice(None) if isinstance(s, BoolTensor) else s for s in slices)
            view = self._getitem_slices(slice_parts)

            def _build_axis_masks(positions):
                masks = []
                for orig_pos, bt in positions:
                    dims_dropped = sum(1 for j in range(orig_pos) if isinstance(slices[j], int))
                    masks.append((orig_pos - dims_dropped, bt._data))
                return masks

            if len(bool_positions) == 1:
                axis, mask_data = _build_axis_masks(bool_positions)[0]
                if isinstance(val, Tensor):
                    view._data.axis_assign(axis, mask_data, val._data)  # type: ignore[arg-type]
                else:
                    view._data.axis_fill(axis, mask_data, self._decay_value(val))  # type: ignore[arg-type]
            else:
                axis_masks = _build_axis_masks(bool_positions)
                if isinstance(val, Tensor):
                    view._data.multi_axis_assign(axis_masks, val._data)  # type: ignore[arg-type]
                else:
                    view._data.multi_axis_fill(axis_masks, self._decay_value(val))  # type: ignore[arg-type]
            return

        # --- IntTensor advanced indexing ---
        if len(int_positions) > 1:
            raise IndexError("Only one IntTensor index per expression is supported")

        slice_parts = tuple(slice(None) if isinstance(s, IntTensor) else s for s in slices)
        view = self._getitem_slices(slice_parts)

        orig_pos, index_tensor = int_positions[0]
        dims_dropped = sum(1 for j in range(orig_pos) if isinstance(slices[j], int))
        axis = orig_pos - dims_dropped

        resolved = _resolve_negative_indices(index_tensor, view.shape[axis])
        if isinstance(val, Tensor):
            view._data.index_assign(axis, resolved._data, val._data)  # type: ignore[arg-type]
        else:
            view._data.index_fill(axis, resolved._data, self._decay_value(val))  # type: ignore[arg-type]

    def _setitem_slices(self, slices, val):
        """Handle setitem with only int/slice components (no BoolTensor)."""
        if len(slices) == 1 and isinstance(slices[0], int):
            self._data._set_element([slices[0]], self._decay_value(val))
        elif len(slices) == 1 and isinstance(slices[0], slice):
            real_slices = [_pyslice_to_slice(slices[0])]
            self._data[real_slices] = val._data
        elif all(isinstance(s, int) for s in slices):
            self._data._set_element(slices, self._decay_value(val))
        else:
            real_slices = [_pyslice_to_slice(s) for s in slices]
            self._data[real_slices] = val._data

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

    def squeeze(self, axis=None):
        if axis is None:
            return self._to_py_tensor(self._data.squeeze())
        return self._to_py_tensor(self._data.squeeze(axis))

    def expand_dims(self, axis):
        return self._to_py_tensor(self._data.expand_dims(axis))

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
