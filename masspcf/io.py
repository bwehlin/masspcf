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

from .tensor import PcfContainerLike, Float32Tensor, Float64Tensor

import masspcf._mpcf_cpp as cpp

def _save(item: PcfContainerLike, file):
    data = item._data
    if isinstance(item, Float32Tensor):
        cpp.IoOps.save_float32_tensor_to_file(data, file)
    else:
        raise TypeError(f'Unsupported tensor type {type(item)}')

def _load(file):
    cpp_tensor = cpp.IoOps.load_tensor_from_file(file)
    if isinstance(cpp_tensor, cpp.Float32Tensor):
        return Float32Tensor(cpp_tensor)
    else:
        raise TypeError(f'File contains unsupported tensor of type {type(cpp_tensor)}')

def save(item: PcfContainerLike, file):
    if isinstance(file, str):
        with open(file, 'wb') as f:
            _save(item, f)
    else:
        _save(item, file)

def load(file):
    if isinstance(file, str):
        with open(file, 'rb') as f:
            return _load(f)
    else:
        return _load(file)

