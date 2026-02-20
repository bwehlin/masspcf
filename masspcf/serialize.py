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
from .tensor import Pcf32Tensor, Pcf64Tensor
from .typing import float32, float64

import numpy as np

def from_serial_content(content : np.ndarray, enumeration : np.ndarray, dtype = None) -> Pcf32Tensor | Pcf64Tensor:
    """Creates a `Tensor` of PCFs from serial numpy data.

    Parameters
    ----------
    content : np.ndarray
        ``(m, 2)`` array of points, where `m` is the sum of lengths of the individual PCFs
    enumeration : np.ndarray
        ``(n_1, n_2, ..., n_k, 2)`` array of `(start, end)` pointers into the content array.
    dtype : datatype
        Sets the ``dtype`` of the resulting PCF ``Array``. If ``None``, uses the ``dtype`` of the supplied ``content`` array. By default, ``None``.

    Returns
    -------
    Tensor
        `Tensor` of shape ``(n_1, n_2, ..., n_k)``, where element ``(i_1, i_2, ..., i_k)`` is a `Pcf` with points ``content[start, stop]`` with ``start=enumeration[i_1,...,i_k, 0]`` and ``stop=enumeration[i_1,...,i_k, 1]``.
    """

    if dtype is None:
        dtype = content.dtype

    if dtype == float32 and content.dtype != np.float32:
        content = content.astype(np.float32)
    elif dtype == float64 and content.dtype != np.float64:
        content = content.astype(np.float64)

    if dtype == float32:
        return Pcf32Tensor(cpp.make_from_serial_content_32(content, enumeration))
    elif dtype == float64:
        return Pcf64Tensor(cpp.make_from_serial_content_64(content, enumeration))

    raise TypeError('Only float32 and float64 dtypes are supported.')

