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
    ts32, ts64,
    ts_pcf32, ts_pcf64,
    ts_barcode32, ts_barcode64,
)


_DTYPE_SUFFIX = {
    ts32: "_f32_f32",
    ts64: "_f64_f64",
    ts_pcf32: "_f32_pcf32",
    ts_pcf64: "_f64_pcf64",
    ts_barcode32: "_f32_barcode32",
    ts_barcode64: "_f64_barcode64",
}


def _cpp_class(prefix, dt):
    suffix = _DTYPE_SUFFIX.get(dt)
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

    def _cpp_for(self, dtype):
        """Return the C++ strategy instance appropriate for *dtype*."""
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

    def _cpp_for(self, dtype):
        cls = _cpp_class("CallableInterpolation", dtype)
        return cls(self._func)
