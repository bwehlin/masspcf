'''
    Copyright 2024 Bjorn Wehlin

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
'''

import matplotlib.pyplot as plt
from .pcf import Pcf
from .array import Container
import numpy as np

def plot(f : Pcf, ax=None, **kwargs):
    ax = plt if ax is None else ax

    def plot_single_(f, **kwargs):
        X = np.array(f)
        ax.step(X[:,0], X[:,1], where='post', **kwargs)

    if isinstance(f, Container):
        if len(f.shape) != 1:
            raise ValueError(f'Expected 1-dimensional array (got array with {f.shape})')
        for i in range(f.shape[0]):
            plot_single_(f[i], label=f'f{i}')
    else:
        plot_single_(f)

