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

from __future__ import annotations

from ..typing import barcode32, barcode64

from .. import _mpcf_cpp as cpp
cpp_p = cpp.persistence

import numpy as np

class Barcode:
    def __init__(self, bc):
        fail = False
        if isinstance(bc, Barcode):
            self._data = bc._data
        elif isinstance(bc, cpp_p.Barcode32):
            self._data = bc
        elif isinstance(bc, cpp_p.Barcode64):
            self._data = bc
        elif isinstance(bc, np.ndarray):
            if bc.dtype == np.float32:
                self._data = cpp_p.Barcode32(bc)
            elif bc.dtype == np.float64:
                self._data = cpp_p.Barcode64(bc)
            else:
                fail = True
        else:
            fail = True

        if isinstance(self._data, cpp_p.Barcode32):
            self.dtype = barcode32
        elif isinstance(self._data, cpp_p.Barcode64):
            self.dtype = barcode64
        else:
            fail = True

        if fail:
            raise TypeError(f'Barcode cannot be constructed with object of type {type(bc)}')

    def __str__(self):
        return self._data.__str__()

    def __repr(self):
        return self._data.__repr__()

    def is_isomorphic_to(self, bc : Barcode):
        return self._data.is_isomorphic_to(bc._data)
