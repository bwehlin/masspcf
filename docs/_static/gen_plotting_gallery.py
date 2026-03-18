"""Generate the plotting gallery figures for the docs."""

import numpy as np
import masspcf as mpcf
from masspcf.random import noisy_sin, noisy_cos
from masspcf.plotting import plot as plotpcf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent


# -- docs snippet start single_pcf --
def plot_single_pcf():
    f = mpcf.Pcf(np.array([[0, 1], [1, 4], [3, 2], [4, 3]], dtype=np.float32))

    fig, ax = plt.subplots(figsize=(5, 2.5))
    plotpcf(f, ax=ax, max_time=6, linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel("f(t)")
    fig.tight_layout()
    return fig
# -- docs snippet end single_pcf --


# -- docs snippet start overlaid --
def plot_overlaid():
    X = noisy_sin((20,), n_points=80)

    fig, ax = plt.subplots(figsize=(5, 2.5))
    plotpcf(X, ax=ax, alpha=0.3, linewidth=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("f(t)")
    ax.set_title("20 noisy sine functions")
    fig.tight_layout()
    return fig
# -- docs snippet end overlaid --


# -- docs snippet start arithmetic --
def plot_arithmetic():
    f = mpcf.Pcf(np.array([[0, 1], [1, 3], [3, 1]], dtype=np.float32))
    g = mpcf.Pcf(np.array([[0, 2], [2, 0]], dtype=np.float32))

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.5), sharex=True, sharey=True)
    for ax, pcf, title in [
        (axes[0], f, "f"),
        (axes[1], g, "g"),
        (axes[2], f + g, "f + g"),
    ]:
        plotpcf(pcf, ax=ax, max_time=5, linewidth=2)
        ax.set_title(title)
        ax.set_ylim(-0.3, 6)
    axes[0].set_ylabel("value")
    fig.tight_layout()
    return fig
# -- docs snippet end arithmetic --


# -- docs snippet start mean_highlight --
def plot_mean_highlight(sin_color="b", cos_color="r"):
    sines = noisy_sin((15,), n_points=100)
    cosines = noisy_cos((15,), n_points=100)

    fig, ax = plt.subplots(figsize=(5, 2.5))
    plotpcf(sines, ax=ax, color=sin_color, linewidth=0.5, alpha=0.2)
    plotpcf(cosines, ax=ax, color=cos_color, linewidth=0.5, alpha=0.2)

    plotpcf(mpcf.mean(sines), ax=ax, color=sin_color, linewidth=2.5, label="mean(sin)")
    plotpcf(mpcf.mean(cosines), ax=ax, color=cos_color, linewidth=2.5, label="mean(cos)")

    ax.set_xlabel("t")
    ax.set_ylabel("f(t)")
    ax.legend()
    fig.tight_layout()
    return fig
# -- docs snippet end mean_highlight --


# -- Generate light and dark variants --
def _save_themed(plot_func, style, bg_color, fg_color, line_color, outfile):
    with plt.style.context(style), \
         plt.rc_context({"axes.facecolor": bg_color,
                         "axes.edgecolor": fg_color,
                         "axes.labelcolor": fg_color,
                         "axes.titlecolor": fg_color,
                         "figure.facecolor": bg_color,
                         "text.color": fg_color,
                         "xtick.color": fg_color,
                         "ytick.color": fg_color,
                         "legend.facecolor": bg_color,
                         "legend.edgecolor": fg_color,
                         "legend.labelcolor": fg_color,
                         "lines.color": line_color}):
        fig = plot_func()
        fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=bg_color)
        plt.close(fig)
        print(f"saved {outfile}")


LIGHT = ("default", "white", "black", "steelblue")
DARK = ("dark_background", "#1a1a2e", "#e0e0e0", "#5dade2")

if __name__ == "__main__":
    gallery = [
        ("gallery_single_pcf", lambda: plot_single_pcf()),
        ("gallery_overlaid", lambda: plot_overlaid()),
        ("gallery_arithmetic", lambda: plot_arithmetic()),
        ("gallery_mean_highlight", lambda: plot_mean_highlight("b", "r")),
    ]
    gallery_dark = [
        ("gallery_single_pcf", lambda: plot_single_pcf()),
        ("gallery_overlaid", lambda: plot_overlaid()),
        ("gallery_arithmetic", lambda: plot_arithmetic()),
        ("gallery_mean_highlight", lambda: plot_mean_highlight("#5dade2", "#ff6b6b")),
    ]
    for (name, func), (_, func_dark) in zip(gallery, gallery_dark):
        _save_themed(func, *LIGHT, HERE / f"{name}_light.png")
        _save_themed(func_dark, *DARK, HERE / f"{name}_dark.png")
