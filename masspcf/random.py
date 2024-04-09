#    Copyright 2024 Bjorn Wehlin
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

from . import mpcf_cpp as cpp
from .pcf import Pcf, tPcf_f32_f32, tPcf_f64_f64
from .array import zeros

from . import typing as dt

def noisy_sin(shape, n_points=20, dtype=dt.float32):
    backend = _get_random_class(dtype)

    A = zeros(shape, dtype=dtype)
    backend.noisy_sin(A.data, n_points)

    return A

def noisy_cos(shape, n_points=20, dtype=dt.float32):
    backend = _get_random_class(dtype)

    A = zeros(shape, dtype=dtype)
    backend.noisy_cos(A.data, n_points)

    return A

def _get_random_class(dtype):
    if dtype == dt.float32:
        return cpp.Random_f32_f32
    elif dtype == dt.float64:
        return cpp.Random_f64_f64
    else:
        raise TypeError('Only float32 and float64 dtypes are supported.')
