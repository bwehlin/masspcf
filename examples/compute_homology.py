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
from masspcf import persistence as mpers

import numpy as np
import math
from tqdm import trange

shape = (10, 20, 30)

min_pcloud_points = 5
max_pcloud_points = 100

pcloud_dim = 4

print(f'Will compute {math.prod(shape)} barcodes...')

pclouds = mpcf.zeros(shape, dtype=mpcf.pcloud64)

for i in trange(shape[0], desc="Initializing data"):
    for j in range(shape[1]):
        for k in range(shape[2]):

            pcloud_shape = (np.random.randint(min_pcloud_points, max_pcloud_points + 1), pcloud_dim)
            pclouds[i, j, k] = np.random.randn(*pcloud_shape)

bcs = mpers.compute_persistent_homology(pclouds)

print(bcs.shape)