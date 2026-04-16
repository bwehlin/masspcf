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

"""Pluggable interpolation for :class:`TimeSeries`.

Nearest and linear interpolation are built in; request them by setting
``TimeSeries(..., interpolation='nearest' | 'linear')``. Linear requires
the value type to support ``+`` and scalar multiplication (scalars and
PCFs qualify; barcodes do not — passing ``'linear'`` for a
barcode-valued series raises).

For anything else — optimal transport on images, Wasserstein matching on
barcodes, cubic splines — implement a batched Python callable and wrap
it in :class:`CallableInterpolation`. Custom strategies are shared_ptr
on the C++ side, so one instance (potentially holding expensive state
like a trained model) can be attached to many time series.
"""

from __future__ import annotations

from .. import _mpcf_cpp as cpp
from ..typing import (
    barcode32, barcode64,
    float32, float64,
    pcf32, pcf64,
)


# Map (TimeSeries-class-tag, element-dtype) -> C++ suffix for looking up
# the right CallableInterpolation_<suffix> class. The class tag lets us
# distinguish scalar-per-timestep TimeSeries (float32 -> _f32_f32) from
# tensor-per-timestep TensorTimeSeries (float32 -> _f32_pcloud32).
_SCALAR_DTYPE_SUFFIX = {
    float32: "_f32_f32",
    float64: "_f64_f64",
    pcf32: "_f32_pcf32",
    pcf64: "_f64_pcf64",
    barcode32: "_f32_barcode32",
    barcode64: "_f64_barcode64",
}

_TENSOR_DTYPE_SUFFIX = {
    float32: "_f32_pcloud32",
    float64: "_f64_pcloud64",
    pcf32: "_f32_pcftensor32",
    pcf64: "_f64_pcftensor64",
    barcode32: "_f32_bctensor32",
    barcode64: "_f64_bctensor64",
}


def _cpp_class(prefix, dt, *, tensor_valued=False):
    suffix_map = _TENSOR_DTYPE_SUFFIX if tensor_valued else _SCALAR_DTYPE_SUFFIX
    suffix = suffix_map.get(dt)
    if suffix is None:
        raise TypeError(f"Unsupported TimeSeries dtype {dt}")
    name = prefix + suffix
    cls = getattr(cpp, name, None)
    if cls is None:
        raise TypeError(f"{prefix} is not available for dtype {dt}")
    return cls


class InterpolationStrategy:
    """Base class for Python-side interpolation strategy wrappers.

    Subclasses are thin adapters that instantiate the appropriate C++
    strategy for a given :class:`TimeSeries` dtype.
    """

    def _cpp_for(self, dtype, *, tensor_valued=False):
        """Return the C++ strategy instance appropriate for *dtype*.

        `tensor_valued` distinguishes ``TimeSeries`` (scalar/Pcf/Barcode
        per timestep) from ``TensorTimeSeries`` (Tensor per timestep).
        """
        raise NotImplementedError


class CallableInterpolation(InterpolationStrategy):
    """Delegates to a user-supplied Python callable (batched).

    The callable receives batched inputs and must return a sequence of
    interpolated values of the same length::

        def my_interp(queries, t_lefts, t_rights, v_lefts, v_rights):
            ...
            return values  # list or numpy array of length len(queries)

    For scalar-valued time series the inputs are Python lists of floats
    (convert with ``np.asarray`` inside the callable if numpy ops are
    desired). For PCF- or barcode-valued series the value inputs are
    lists of :class:`Pcf` / :class:`Barcode` C++ wrapper objects.
    """

    def __init__(self, func):
        if not callable(func):
            raise TypeError("CallableInterpolation requires a callable")
        self._func = func

    def _cpp_for(self, dtype, *, tensor_valued=False):
        cls = _cpp_class("CallableInterpolation", dtype,
                         tensor_valued=tensor_valued)
        return cls(self._func)
