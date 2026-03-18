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


# -- docs snippet start barcode --
def plot_barcode_example(h0_color="steelblue", h1_color="orangered"):
    from masspcf.persistence import Barcode
    from masspcf.plotting import plot_barcode

    bc_h0 = Barcode(np.array([
        [0.0, np.inf], [0.0, 1.8], [0.0, 0.6], [0.1, 1.2],
    ], dtype=np.float64))

    bc_h1 = Barcode(np.array([
        [0.5, 2.0], [0.8, 1.5], [1.0, 3.0],
    ], dtype=np.float64))

    fig, ax = plt.subplots(figsize=(5, 2.5))
    y = plot_barcode(bc_h0, ax=ax, color=h0_color, linewidth=2, label="H0")
    plot_barcode(bc_h1, ax=ax, y_offset=y + 1, color=h1_color, linewidth=2, label="H1")
    ax.set_xlabel("t")
    ax.set_yticks([])
    ax.legend()
    fig.tight_layout()
    return fig
# -- docs snippet end barcode --


# -- docs snippet start tda_pipeline --
def plot_tda_pipeline(h0_color="steelblue", h1_color="orangered"):
    from masspcf import persistence as mpers
    from masspcf.plotting import plot_barcode

    # 1. Noisy circle (clear H1 topology)
    rng = np.random.RandomState(10)
    theta = rng.uniform(0, 2 * np.pi, 30)
    r = 1.0 + rng.normal(0, 0.15, 30)
    points = np.column_stack([r * np.cos(theta), r * np.sin(theta)]).astype(np.float64)

    # 2. Compute persistent homology
    bcs = mpers.compute_persistent_homology(points, maxDim=1, verbose=False)
    bc_h0, bc_h1 = bcs[0], bcs[1]

    # 3. Convert to stable rank
    sranks = mpers.barcode_to_stable_rank(bcs)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3),
                                         gridspec_kw={"width_ratios": [1, 1, 1.2]})

    # Left: point cloud
    ax1.scatter(points[:, 0], points[:, 1], s=15, color="grey", edgecolors="black",
                linewidths=0.5)
    ax1.set_aspect("equal")
    ax1.set_title("Point cloud")

    # Middle: persistence diagram (via persim)
    import persim
    persim.plot_diagrams(
        [np.asarray(bc_h0), np.asarray(bc_h1)],
        ax=ax2, legend=True, show=False,
    )
    legend = ax2.get_legend()
    legend.get_frame().set_alpha(0)
    fg = ax2.xaxis.label.get_color()
    for text in legend.get_texts():
        text.set_color(fg)
    for line in ax2.get_lines():
        line.set_color(fg)
    ax2.set_title("Persistence diagram")

    # Right: stable rank
    plotpcf(sranks[0], ax=ax3, max_time=2, color=h0_color, linewidth=2,
            label="H0")
    plotpcf(sranks[1], ax=ax3, max_time=2, color=h1_color, linewidth=2,
            label="H1")
    ax3.set_xlabel("t")
    ax3.set_ylabel("rank")
    ax3.set_title("Stable rank")
    leg3 = ax3.legend(fontsize=8)
    leg3.get_frame().set_alpha(0)
    for text in leg3.get_texts():
        text.set_color(fg)

    fig.tight_layout(w_pad=1.5)
    return fig
# -- docs snippet end tda_pipeline --


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
        ("gallery_barcode", lambda: plot_barcode_example("steelblue", "orangered")),
        ("gallery_tda_pipeline", lambda: plot_tda_pipeline("steelblue", "orangered")),
    ]
    gallery_dark = [
        ("gallery_single_pcf", lambda: plot_single_pcf()),
        ("gallery_overlaid", lambda: plot_overlaid()),
        ("gallery_arithmetic", lambda: plot_arithmetic()),
        ("gallery_mean_highlight", lambda: plot_mean_highlight("#5dade2", "#ff6b6b")),
        ("gallery_barcode", lambda: plot_barcode_example("#5dade2", "#ff6b6b")),
        ("gallery_tda_pipeline", lambda: plot_tda_pipeline("#5dade2", "#ff6b6b")),
    ]
    for (name, func), (_, func_dark) in zip(gallery, gallery_dark):
        _save_themed(func, *LIGHT, HERE / f"{name}_light.png")
        _save_themed(func_dark, *DARK, HERE / f"{name}_dark.png")
