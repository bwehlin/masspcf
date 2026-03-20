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

import numpy as np

# mpcf type aliases

class float32:
    pass


class float64:
    pass


class int32:
    pass


class int64:
    pass


class pcf32:
    pass


class pcf64:
    pass


class pcf32i:
    pass


class pcf64i:
    pass


class pcloud32:
    pass


class pcloud64:
    pass


class barcode32:
    pass


class barcode64:
    pass


class symmat32:
    pass


class symmat64:
    pass


class distmat32:
    pass


class distmat64:
    pass


class boolean:
    pass


Dtype = type[
    pcf32 | pcf64 | pcf32i | pcf64i
    | float32 | float64
    | int32 | int64
    | pcloud32 | pcloud64
    | barcode32 | barcode64
    | symmat32 | symmat64
    | distmat32 | distmat64
    | boolean
]

_MPCF_TO_NP = {
    float32: np.float32,
    float64: np.float64,
    int32: np.int32,
    int64: np.int64,
}

_NP_TO_MPCF = {
    np.float32: float32,
    np.float64: float64,
    np.int32: int32,
    np.int64: int64,
}


def _assert_valid_dtype(dtype, valid_dtypes):
    def name_of(dt):
        if isinstance(dt, type):
            return dt.__name__
        else:
            return str(dt)

    if not any(dtype == valid_dtype for valid_dtype in valid_dtypes):
        raise TypeError(
            "Only the following dtypes are supported: "
            + ", ".join(name_of(valid_dtype) for valid_dtype in valid_dtypes)
            + f" (supplied {name_of(dtype)})"
        )


def _validate_dtype(dtype, valid_dtypes):
    _assert_valid_dtype(dtype, valid_dtypes)
    return dtype