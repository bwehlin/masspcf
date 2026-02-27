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

from .._tensor_base import Tensor
from ..typing import barcode32, barcode64
from .barcode import Barcode

import numpy as np

from .. import _mpcf_cpp as cpp
cpp_p = cpp.persistence

class BarcodeTensor(Tensor):
    def __init__(self):
        super().__init__()

    def _decay_value(self, val):
        return val._data

    def _represent_element(self, element):
        return Barcode(element)

    def _get_valid_setitem_dtypes(self):
        return [Barcode, np.ndarray]

class Barcode32Tensor(BarcodeTensor):
    def __init__(self, data : cpp_p.Barcode32Tensor):
        super().__init__()
        self._data = data
        self.dtype = barcode32

    def _to_py_tensor(self, data):
        return Barcode32Tensor(data)

class Barcode64Tensor(BarcodeTensor):
    def __init__(self, data : cpp_p.Barcode64Tensor):
        super().__init__()
        self._data = data
        self.dtype = barcode64

    def _to_py_tensor(self, data):
        return Barcode64Tensor(data)
