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

from .persistence.barcode import Barcode
from .persistence.ph_tensor import BarcodeTensor
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


def plot_barcode(bc, ax=None, y_offset=0, **kwargs):
    """Plot a persistence barcode as horizontal line segments.

    Each bar is drawn as a horizontal segment from birth to death.
    Bars with infinite death are drawn as arrows extending to the right
    edge of the plot.

    Parameters
    ----------
    bc : Barcode or BarcodeTensor
        A single ``Barcode`` or a 1-D ``BarcodeTensor``. For a tensor,
        the barcodes are stacked vertically in order.
    ax : matplotlib axes, optional
        Axes to plot on. If ``None``, uses ``matplotlib.pyplot`` directly.
    y_offset : int, optional
        Starting y position for the first bar. Useful when stacking
        multiple barcodes on the same axes.
    **kwargs
        Additional keyword arguments passed to
        ``matplotlib.collections.LineCollection``
        (e.g. ``color``, ``linewidth``, ``alpha``, ``label``).

    Returns
    -------
    int
        The next available y position (for stacking).
    """
    from matplotlib.collections import LineCollection

    ax = plt.gca() if ax is None else ax

    if isinstance(bc, BarcodeTensor):
        if len(bc.shape) != 1:
            raise ValueError(f"Expected 1-dimensional tensor (got shape {bc.shape})")
        y = y_offset
        for i in range(bc.shape[0]):
            y = plot_barcode(bc[i], ax=ax, y_offset=y, **kwargs)
        return y

    bars = np.asarray(bc)
    if len(bars) == 0:
        return y_offset

    finite_mask = np.isfinite(bars[:, 1])
    finite_bars = bars[finite_mask]
    inf_bars = bars[~finite_mask]

    # Draw finite bars
    defaults = {"linewidth": 1.5}
    defaults.update(kwargs)

    if len(finite_bars) > 0:
        finite_indices = np.where(finite_mask)[0]
        segments = [[(b, y_offset + i), (d, y_offset + i)]
                    for i, (b, d) in zip(finite_indices, finite_bars)]
        lc = LineCollection(segments, **defaults)
        ax.add_collection(lc)

    # Draw infinite bars: line segment extending to the right edge with arrowhead
    if len(inf_bars) > 0:
        inf_indices = np.where(~finite_mask)[0]
        color = defaults.get("color", None)
        lw = defaults.get("linewidth", 1.5)

        # Determine xmax from all finite deaths plus margin
        if len(finite_bars) > 0:
            xmax = finite_bars[:, 1].max() * 1.15
        else:
            xmax = inf_bars[:, 0].max() + 1.0

        inf_segs = [[(b, y_offset + i), (xmax, y_offset + i)]
                    for i, (b, _) in zip(inf_indices, inf_bars)]
        inf_lc = LineCollection(inf_segs, linewidth=lw, color=color)
        ax.add_collection(inf_lc)
        for i in inf_indices:
            ax.annotate("",
                xy=(xmax, y_offset + i),
                xytext=(xmax - (xmax * 0.02), y_offset + i),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))

    ax.autoscale_view()

    return y_offset + len(bars)
