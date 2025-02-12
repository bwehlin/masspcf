#    Copyright 2024-2025 Bjorn Wehlin
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

import matplotlib.pyplot as plt
from .pcf import Pcf
from .array import Array, View, Container, max_time
import numpy as np
from typing import Union

def plot(f : Union[Pcf, Array, View, Container], fmt='', ax=None, auto_label=False, **kwargs):
    ax = plt if ax is None else ax

    def plot_single_(f, maxtime, **kwargs1):
        X = np.array(f)
        if maxtime is not None and X[-1,0] != maxtime:
            X = np.vstack((X, [maxtime, X[-1,1]]))
        ax.step(X[:,0], X[:,1], fmt, where='post', **kwargs, **kwargs1)

    if isinstance(f, Container):
        if len(f.shape) != 1:
            raise ValueError(f'Expected 1-dimensional array (got array with {f.shape})')
        for i in range(f.shape[0]):
            kw = {'label': f'f{i}'} if auto_label else {}
            plot_single_(f[i], max_time(f), **kw)
    else:
        plot_single_(f, None)

