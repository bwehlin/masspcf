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

import warnings
import numpy as np

# mpcf type aliases

f32 = np.float32
f64 = np.float64

class pcf32:
    pass

class pcf64:
    pass

class pcloud32:
    pass

class pcloud64:
    pass

class _DeprecatedDtype:
    def __init__(self, name, replacement, standin):
        self._name = name
        self._replacement = replacement
        self._standin = standin

    def __repr__(self):
        return self._name

    def __str__(self):
        return self._name

    def __name__(self):
        return self.name

    def show_deprectation_warning(self):
        warnings.warn(
            f"{self._name} is deprecated; use {self._replacement} instead.",
            DeprecationWarning,
            stacklevel=2
        )

    def get_standin(self):
        self.show_deprectation_warning()
        return self._standin

def _check_deprecated_dtype(dtype):
    if isinstance(dtype, _DeprecatedDtype):
        return dtype.get_standin()
    else:
        return dtype

def _assert_valid_dtype(dtype, valid_dtypes):
    if not any(dtype == valid_dtype for valid_dtype in valid_dtypes):
        raise TypeError("Only the following dtypes are supported: " + ", ".join(valid_dtype.__name__ for valid_dtype in valid_dtypes) + f" (supplied {dtype.__name__})")

def _validate_dtype(dtype, valid_dtypes):
    dtype = _check_deprecated_dtype(dtype)
    _assert_valid_dtype(dtype, valid_dtypes)
    return dtype

float32 = _DeprecatedDtype("float32", "pcf32/pcf64 for 32/64-bit float PCFs and f32/f64 for numeric 32/64-bit floats", pcf32)
float64 = _DeprecatedDtype("float64", "pcf32/pcf64 for 32/64-bit float PCFs and f32/f64 for numeric 32/64-bit floats", pcf64)
