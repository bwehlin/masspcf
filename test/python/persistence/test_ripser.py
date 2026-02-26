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
import masspcf.persistence as mpers

import numpy as np
def test_persistence_ripser_compute_euclidean_barcodes_returns_correct_dtype_and_shape():
    Xnp = np.random.randn(10, 2)
    X = mpcf.DoubleTensor(Xnp)

    bcs = mpers.compute_barcodes_euclidean_pcloud_ripser(X, maxDim=3)

    assert isinstance(bcs, mpers.BarcodeTensor)
    assert bcs.shape == (4,)



"""
def test_123():
    X = mpcf.zeros((100, 1000))

    X[0, 0] = ...

    Y = mpers.compute_persistence(X, maxDim=1)

    assert Y.shape == (100, 1000, 2)

"""