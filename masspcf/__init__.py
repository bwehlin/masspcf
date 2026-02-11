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


from .pcf import Pcf, average
#from .array import Array, View, Shape, Container, ContainerLike, zeros, mean, from_serial_content
from .matrix_computations import pdist, l2_kernel
from .norms import l1_norm, l2_norm, lp_norm, linfinity_norm
from .typing import float32, float64
from .tensor import FloatTensor, DoubleTensor, Pcf32Tensor, Pcf64Tensor, TShape, zerosT, zeros

from . import system

__all__ = ['pcf', 'plotting']
