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
import pytest

def test_assign_tensor_to_slice_1d():
    X0 = mpcf.random.noisy_sin((10, ))
    X1 = mpcf.random.noisy_sin((10, ))

    A = mpcf.zeros((2, 10))

    A[0, :] = X0
    A[1, :] = X1

    assert A[0, :] == X0
    assert A[1, :] == X1

def test_assign_tensor_to_slice_nd():
    X0 = mpcf.random.noisy_sin((10, 10))
    X1 = mpcf.random.noisy_sin((10, 20))

    A = mpcf.zeros((10, 30))

    A[:, 0:10] = X0
    A[:, 10:30] = X1

    assert A[:, 0:10] == X0
    assert A[:, 10:30] == X1

def test_assign_tensor_to_slice_incommensurate_dims():
    X0 = mpcf.random.noisy_sin((10, 10))

    A = mpcf.zeros((10, 30))

    with pytest.raises(RuntimeError):
        A[:, 0:5] = X0
