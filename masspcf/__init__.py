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


from . import random, system
from ._tensor_base import Shape
from .distance import pdist
from .io import load, save
from .norms import lp_norm
from .pcf import Pcf
from .reductions import max_time, mean
from .serialize import from_serial_content
from .tensor import (
    Float32Tensor,
    Float64Tensor,
    Pcf32Tensor,
    Pcf64Tensor,
    PointCloud32Tensor,
    PointCloud64Tensor,
)
from .tensor_create import zeros
from .typing import (
    barcode32,
    barcode64,
    f32,
    f64,
    float32,  # deprecated
    float64,
    pcf32,
    pcf64,
    pcloud32,
    pcloud64,
)
