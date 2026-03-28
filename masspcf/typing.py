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
    """32-bit floating-point scalar dtype."""
    pass


class float64:
    """64-bit floating-point scalar dtype."""
    pass


class int32:
    """32-bit integer scalar dtype."""
    pass


class int64:
    """64-bit integer scalar dtype."""
    pass


class uint32:
    """32-bit unsigned integer scalar dtype."""
    pass


class uint64:
    """64-bit unsigned integer scalar dtype."""
    pass


class pcf32:
    """32-bit piecewise constant function dtype."""
    pass


class pcf64:
    """64-bit piecewise constant function dtype."""
    pass


class pcf32i:
    """32-bit integer piecewise constant function dtype."""
    pass


class pcf64i:
    """64-bit integer piecewise constant function dtype."""
    pass


class pcloud32:
    """32-bit point cloud dtype."""
    pass


class pcloud64:
    """64-bit point cloud dtype."""
    pass


class barcode32:
    """32-bit persistence barcode dtype."""
    pass


class barcode64:
    """64-bit persistence barcode dtype."""
    pass


class symmat32:
    """32-bit symmetric matrix dtype."""
    pass


class symmat64:
    """64-bit symmetric matrix dtype."""
    pass


class distmat32:
    """32-bit distance matrix dtype."""
    pass


class distmat64:
    """64-bit distance matrix dtype."""
    pass


class boolean:
    """Boolean dtype."""
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