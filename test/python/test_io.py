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

import io

import numpy as np

import masspcf as mpcf


def test_float32_tensor_roundtrip():
    original = mpcf.Float32Tensor(np.random.randn(2, 3))

    buf = io.BytesIO()
    mpcf.save(original, buf)

    buf.seek(0)  # rewind before reading!
    restored = mpcf.load(buf)

    assert original == restored
