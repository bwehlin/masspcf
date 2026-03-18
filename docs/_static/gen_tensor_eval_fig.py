"""Generate the tensor evaluation example figures for the docs."""

import numpy as np
import masspcf as mpcf
from masspcf.plotting import plot as plotpcf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent

# -- PCFs used in the docs example --
# f(t) = 1 on [0,1), 4 on [1,3), 2 on [3,inf)
f = mpcf.Pcf(np.array([[0, 1], [1, 4], [3, 2]], dtype=np.float32))

# g(t) = 3 on [0,2), 1 on [2,inf)
g = mpcf.Pcf(np.array([[0, 3], [2, 1]], dtype=np.float32))

X = mpcf.zeros((2, 2))
X[0, 0] = f
X[0, 1] = g
X[1, 0] = g
X[1, 1] = f

XLIM = (-0.2, 5)
YLIM = (-0.3, 5)
T_EVAL = 2


# -- included in docs via literalinclude :pyobject: --
def plot_pcf_definitions():
    fig, (ax_f, ax_g) = plt.subplots(1, 2, figsize=(6, 2.5),
                                      sharex=True, sharey=True)
    plotpcf(f, ax=ax_f, linewidth=2)
    ax_f.set_title("f")
    ax_f.set_xlim(*XLIM)
    ax_f.set_ylim(*YLIM)

    plotpcf(g, ax=ax_g, linewidth=2)
    ax_g.set_title("g")
    ax_g.set_xlim(*XLIM)
    ax_g.set_ylim(*YLIM)

    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_tensor_eval_example():
    fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True, sharey=True)

    for ax, pcf, label in [
        (axes[0, 0], f, "X[0,0] = f"),
        (axes[0, 1], g, "X[0,1] = g"),
        (axes[1, 0], g, "X[1,0] = g"),
        (axes[1, 1], f, "X[1,1] = f"),
    ]:
        plotpcf(pcf, ax=ax, linewidth=2)
        val = pcf(T_EVAL)
        ax.axvline(T_EVAL, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.plot(T_EVAL, val, "ro", markersize=6, zorder=5)
        ax.set_title(label, fontsize=10)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)

    fig.suptitle(f"X({T_EVAL}) = {X(T_EVAL).tolist()}", fontsize=11)
    fig.tight_layout()
    return fig


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
                         "lines.color": line_color}):
        fig = plot_func()
        fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=bg_color)
        plt.close(fig)
        print(f"saved {outfile}")


LIGHT = ("default", "white", "black", "steelblue")
DARK = ("dark_background", "#1a1a2e", "#e0e0e0", "#5dade2")

if __name__ == "__main__":
    for name, plot_func in [("pcf_definitions", plot_pcf_definitions),
                             ("tensor_eval_example", plot_tensor_eval_example)]:
        _save_themed(plot_func, *LIGHT, HERE / f"{name}_light.png")
        _save_themed(plot_func, *DARK, HERE / f"{name}_dark.png")
