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

from . import _mpcf_cpp as cpp
from .tensor import PcfContainerLike, Pcf32Tensor, Pcf64Tensor
from .pcf import Pcf

def _get_tensor_and_backend(fs):
    if isinstance(fs, Pcf32Tensor):
        print('ii32')
        return fs, cpp.Reductions_f32_f32
    elif isinstance(fs, Pcf64Tensor):
        print('ii64')
        return fs, cpp.Reductions_f64_f64
    else:
        raise ValueError('Unsupported input type.')

def _to_tensor_or_pcf(outFs):
    if isinstance(outFs, cpp.Pcf32Tensor) or isinstance(outFs, cpp.Pcf64Tensor):
        if len(outFs.shape) == 1 and outFs.shape[0] == 1:
            return Pcf(outFs._get_element(0))

    if isinstance(outFs, cpp.Pcf32Tensor):
        return Pcf32Tensor(outFs)
    elif isinstance(outFs, cpp.Pcf64Tensor):
        return Pcf64Tensor(outFs)
    else:
        raise ValueError('Invalid output type (this is probably a bug -- please report it!).')

def mean(fs : PcfContainerLike, dim : int=0):
    tensor, backend = _get_tensor_and_backend(fs)
    print(tensor)
    print(backend)
    return _to_tensor_or_pcf(backend.mean(tensor._data, dim))

