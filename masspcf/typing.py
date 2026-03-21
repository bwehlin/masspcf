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


class uint32:
    pass


class uint64:
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
    | int32 | int64 | uint32 | uint64
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
    uint32: np.uint32,
    uint64: np.uint64,
}

_NP_TO_MPCF = {
    np.float32: float32,
    np.float64: float64,
    np.int32: int32,
    np.int64: int64,
    np.uint32: uint32,
    np.uint64: uint64,
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


# Short tags used to look up C++ cast functions by naming convention:
# cpp.cast_{src_tag}_{dst_tag}
_DTYPE_TAG = {
    float32: "f32", float64: "f64",
    int32: "i32", int64: "i64",
    uint32: "u32", uint64: "u64",
    pcf32: "pcf32", pcf64: "pcf64",
    pcf32i: "pcf32i", pcf64i: "pcf64i",
    pcloud32: "pcloud32", pcloud64: "pcloud64",
}


# Populated lazily to avoid circular imports (tensor.py -> typing.py)
_DTYPE_TO_WRAPPER = {}


def _init_dtype_wrappers():
    if _DTYPE_TO_WRAPPER:
        return
    from .tensor import FloatTensor, IntTensor, PcfTensor, IntPcfTensor, PointCloudTensor
    _DTYPE_TO_WRAPPER.update({
        float32: FloatTensor, float64: FloatTensor,
        int32: IntTensor, int64: IntTensor,
        uint32: IntTensor, uint64: IntTensor,
        pcf32: PcfTensor, pcf64: PcfTensor,
        pcf32i: IntPcfTensor, pcf64i: IntPcfTensor,
        pcloud32: PointCloudTensor, pcloud64: PointCloudTensor,
    })