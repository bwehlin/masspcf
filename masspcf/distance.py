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
from .tensor import PcfContainerLike, _to_tensor_pcf, _get_backend, TensorType
from .typing import float32, float64
from .np_support import numpy_type
from .async_task import _wait_for_task

import numpy as np

def _get_distance_backend(fs) -> cpp.Distance_f32_f32 | cpp.Distance_f64_f64:
    mapping = { (TensorType.PCF, float32): cpp.Distance_f32_f32,
                (TensorType.PCF, float64): cpp.Distance_f64_f64 }

    return _get_backend(fs, mapping)

def pdist(fs : PcfContainerLike, p=1, verbose=True):
    X = _to_tensor_pcf(fs)

    if len(X.shape) != 1:
        raise ValueError('1d tensor expected.')

    backend = _get_distance_backend(fs)
    matrix = np.zeros((X.shape[0], X.shape[0]), dtype=numpy_type(X))

    task = None
    try:
        task = backend.pdist_l1(matrix, X._data)
        _wait_for_task(task, verbose=verbose)
        return matrix
    finally:
        if task is not None:
            task.request_stop()
            _wait_for_task(task, verbose=verbose)

