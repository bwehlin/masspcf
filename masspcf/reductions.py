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

from . import _mpcf_cpp as cpp
from .functional.pcf import Pcf
from .tensor import (
    FloatTensor,
    PcfTensor,
    PcfContainerLike,
    _get_backend,
)
from .typing import pcf32, pcf64


def _get_tensor_and_backend(fs):
    mapping = {pcf32: cpp.Reductions_f32_f32, pcf64: cpp.Reductions_f64_f64}
    backend, tensor = _get_backend(fs, mapping)
    return tensor, backend


def _to_tensor_or_val(outFs):
    if isinstance(outFs, (cpp.Pcf32Tensor, cpp.Pcf64Tensor)):
        if len(outFs.shape) == 1 and outFs.shape[0] == 1:
            return Pcf(outFs._get_element(0))
        return PcfTensor(outFs)
    elif isinstance(outFs, (cpp.Float32Tensor, cpp.Float64Tensor)):
        if len(outFs.shape) == 1 and outFs.shape[0] == 1:
            return outFs._get_element(0)
        return FloatTensor(outFs)
    else:
        raise ValueError(
            "Invalid output type (this is probably a bug -- please report it!)."
        )


def mean(fs: PcfContainerLike, dim: int = 0):
    r"""Compute the pointwise mean of a PCF tensor along the given dimension.

    The mean is computed pointwise in time: for functions
    :math:`f_1, f_2, \ldots, f_n` being reduced, the resulting function
    :math:`\bar{f}` satisfies

    .. math::
        \bar{f}(t) = \frac{1}{n} \sum_{i=1}^{n} f_i(t)

    for all :math:`t`.

    See :ref:`tensors-how-dim-works` for a detailed explanation of dimension
    reduction semantics.

    Parameters
    ----------
    fs : PcfContainerLike
        A ``PcfTensor`` with dtype ``pcf32`` or ``pcf64``.
    dim : int, optional
        Dimension along which to reduce, by default 0.

    Returns
    -------
    Pcf or PcfTensor
        A single ``Pcf`` if the result is scalar, otherwise a ``PcfTensor``
        with the reduced dimension removed.
    """
    tensor, backend = _get_tensor_and_backend(fs)
    return _to_tensor_or_val(backend.mean(tensor._data, dim))


def max_time(fs: PcfContainerLike, dim: int = 0):
    r"""Compute the maximum breakpoint time along the given dimension.

    For each PCF :math:`f_i` with breakpoints
    :math:`(t_0^{(i)}, t_1^{(i)}, \ldots, t_{n_i-1}^{(i)})`, let
    :math:`T_i = t_{n_i-1}^{(i)}` be the last breakpoint. For functions
    :math:`f_1, f_2, \ldots, f_n` being reduced, this returns

    .. math::
        \max(T_1, T_2, \ldots, T_n).

    The result is numeric, not a PCF.

    See :ref:`tensors-how-dim-works` for a detailed explanation of dimension
    reduction semantics.

    Parameters
    ----------
    fs : PcfContainerLike
        A ``PcfTensor`` with dtype ``pcf32`` or ``pcf64``.
    dim : int, optional
        Dimension along which to reduce, by default 0.

    Returns
    -------
    float or FloatTensor
        A scalar if the result is 0-dimensional, otherwise a numeric tensor.
    """
    tensor, backend = _get_tensor_and_backend(fs)
    return _to_tensor_or_val(backend.max_time(tensor._data, dim))
