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
from . import BarcodeTensor
from ..tensor import (FloatTensor, DoubleTensor,
                      _get_backend
                      )

from ..typing import f32, f64
from .tensor import BarcodeTensor

from .._mpcf_cpp import persistence as cpp_p

def compute_barcodes_euclidean_pcloud_ripser(X : FloatTensor | DoubleTensor):
    backend = _get_backend(X, {
        f32 : cpp_p.PersistenceRipser32,
        f64 : cpp_p.PersistenceRipser64
    })

    out = backend.compute_barcodes_euclidean_pcloud_ripser(X._data)

    if isinstance(X, FloatTensor):
        return Bar(out)

