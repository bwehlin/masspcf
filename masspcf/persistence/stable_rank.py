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

from .. import _mpcf_cpp as cpp
cpp_p = cpp.persistence

from ..typing import barcode32, barcode64
from ..tensor import _get_backend
from ..pcf import Pcf
from .barcode import Barcode

def barcode_to_stable_rank(bc : Barcode):

    backend, X = _get_backend(bc, {
        barcode32 : cpp_p.PersistenceStableRank32,
        barcode64 : cpp_p.PersistenceStableRank64
    })

    return Pcf(backend.barcode_to_stable_rank(X._data))
