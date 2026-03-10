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

import masspcf as mpcf
import masspcf._mpcf_cpp as mcpp

import numpy as np

def test_numpy_tensor_create_gives_correct_cpp_type():
    Xnp = np.zeros((10, 20), dtype=np.float32)
    X = mpcf.Float32Tensor(Xnp)
    assert isinstance(X._data, mcpp.Float32Tensor)

    Xnp = np.zeros((10, 20), dtype=np.float64)
    X = mpcf.Float64Tensor(Xnp)
    assert isinstance(X._data, mcpp.Float64Tensor)
