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


import matplotlib.pyplot as plt
import numpy as np

from .reductions import max_time as max_time_reduction
from .tensor import PcfContainerLike, PcfTensor


def plot(f: PcfContainerLike, fmt="", ax=None, auto_label=False, max_time=None, **kwargs):
    """Plot one or more PCFs using matplotlib's step function.

    Parameters
    ----------
    f : PcfContainerLike
        A single ``Pcf`` or a 1-D ``PcfTensor``.
    fmt : str, optional
        A matplotlib format string (e.g. ``'r--'``), by default ``''``.
    ax : matplotlib axes, optional
        Axes to plot on. If ``None``, uses ``matplotlib.pyplot`` directly.
    auto_label : bool, optional
        If ``True`` and ``f`` is a tensor, label each PCF as ``f0``, ``f1``,
        etc. By default ``False``.
    max_time : float, optional
        Extend the plot so the final constant segment reaches this time.
        If ``None``, single PCFs are not extended and tensors extend to
        the latest breakpoint across all elements.
    **kwargs
        Additional keyword arguments passed to ``matplotlib.pyplot.step``
        (e.g. ``color``, ``linewidth``, ``alpha``, ``label``).

    Raises
    ------
    ValueError
        If ``f`` is a tensor with more than one dimension.
    """
    ax = plt if ax is None else ax

    def plot_single_(f, maxtime, **kwargs1):
        X = np.array(f)
        if maxtime is not None and X[-1, 0] != maxtime:
            X = np.vstack((X, [maxtime, X[-1, 1]]))
        ax.step(X[:, 0], X[:, 1], fmt, where="post", **kwargs, **kwargs1)

    if isinstance(f, PcfTensor):
        if len(f.shape) != 1:
            raise ValueError(f"Expected 1-dimensional array (got array with {f.shape})")
        mt = max_time if max_time is not None else np.array(max_time_reduction(f))
        for i in range(f.shape[0]):
            kw = {"label": f"f{i}"} if auto_label else {}
            plot_single_(f[i], mt, **kw)
    else:
        plot_single_(f, max_time)
