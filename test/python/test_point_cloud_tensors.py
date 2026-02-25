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

import numpy as np

def test_can_create_2_point_clouds():
    X = mpcf.zeros((2,), dtype=mpcf.pcloud64)

    print(type(X))

    X[0] = np.random.randn(10, 2)
    X[1] = np.random.randn(20, 2)

    assert X[0].shape == (10, 2)
    assert X[1].shape == (20, 2)
