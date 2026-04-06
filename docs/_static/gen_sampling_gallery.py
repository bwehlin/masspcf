"""Generate the sampling gallery figures for the docs."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from pathlib import Path

import masspcf as mpcf
from masspcf.point_process import sample_poisson
from masspcf.random import Generator

HERE = Path(__file__).parent

# Reproducible dataset via Poisson point process
gen = Generator(seed=42)
_pc_tensor = sample_poisson((1,), dim=2, rate=10.0,
                            lo=np.array([-2.0, -2.0]),
                            hi=np.array([5.0, 5.0]),
                            generator=gen)
SOURCE_PTS = np.asarray(_pc_tensor[0])
VANTAGE = np.array([0.0, 0.0])

# ---------------------------------------------------------------------------
# Okabe-Ito CVD-safe palette (https://jfly.uni-koeln.de/color/)
# ---------------------------------------------------------------------------
OI_SKY_BLUE = "#56B4E9"
OI_VERMILLION = "#D55E00"
OI_ORANGE = "#E69F00"
OI_BLUE = "#0072B2"


def _weight_from_dist(pts, vantage, dist_type, **kwargs):
    """Compute per-point weights for a given distribution type."""
    dists = np.linalg.norm(pts - vantage, axis=1)
    if dist_type == "gaussian":
        mu, sigma = kwargs["mean"], kwargs["sigma"]
        return np.exp(-0.5 * ((dists - mu) / sigma) ** 2)
    elif dist_type == "uniform":
        lo, hi = kwargs["lo"], kwargs["hi"]
        return np.where((dists >= lo) & (dists <= hi), 1.0, 0.0)
    elif dist_type == "mixture":
        weights_out = np.zeros(len(dists))
        for comp, w in zip(kwargs["components"], kwargs["weights"]):
            weights_out += w * _weight_from_dist(pts, vantage, comp["type"], **comp["params"])
        return weights_out
    raise ValueError(f"Unknown dist_type: {dist_type}")


def _sample_indices(weights, k, rng):
    """Simple weighted sampling with replacement."""
    probs = weights / weights.sum()
    return rng.choice(len(weights), size=k, replace=True, p=probs)


def _plot_sampling_row(dist_type, dist_kwargs, dist_label,
                       source_color=OI_SKY_BLUE,
                       sample_color=OI_VERMILLION,
                       vantage_color="black",
                       vantage_edge="white",
                       zero_color="#bbbbbb",
                       cmap_name="viridis",
                       cmap_range=(0.0, 0.85)):
    """Create a 1x3 figure: weights | sample 1 | sample 2.

    Design choices following CVD-safe and visibility guidelines:
    - Okabe-Ito palette for categorical distinctions (source vs sampled)
    - viridis colormap (perceptually uniform, CVD-safe) for weight heatmap
    - Shape redundancy: vantage uses star marker, source/sampled use circles
    - Size hierarchy: vantage > sampled > source for clear layering
    - Edge strokes on all markers for figure-ground separation
    """
    from matplotlib.gridspec import GridSpec

    pts = SOURCE_PTS
    vantage = VANTAGE
    weights = _weight_from_dist(pts, vantage, dist_type, **dist_kwargs)

    # Extra width for the first panel to accommodate the colorbar
    fig = plt.figure(figsize=(12, 3.8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 1, 1], wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # -- Panel 1: all points colored by weight --
    ax = axes[0]
    zero_mask = weights < 1e-12
    if zero_mask.any():
        ax.scatter(pts[zero_mask, 0], pts[zero_mask, 1],
                   s=18, color=zero_color, alpha=0.5, zorder=1,
                   edgecolors="white", linewidths=0.3)
    nonzero = ~zero_mask
    if nonzero.any():
        # Truncate colormap to avoid light extremes that vanish on white/dark bg
        base_cmap = plt.get_cmap(cmap_name)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "trunc", base_cmap(np.linspace(cmap_range[0], cmap_range[1], 256)))
        norm = Normalize(vmin=0, vmax=weights[nonzero].max())
        ax.scatter(pts[nonzero, 0], pts[nonzero, 1],
                   c=weights[nonzero], cmap=cmap, norm=norm,
                   s=28, alpha=0.9, zorder=2,
                   edgecolors="white", linewidths=0.3)
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                            ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("weight $g(d)$", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    ax.scatter(*vantage, s=180, color=vantage_color, marker="*",
               zorder=5, edgecolors=vantage_edge, linewidths=0.8)
    ax.set_title(f"Weight: {dist_label}", fontsize=10)

    # -- Panels 2-3: two independent samples --
    for panel_idx, seed in enumerate([123, 456], start=1):
        ax = axes[panel_idx]
        rng = np.random.RandomState(seed)
        k = 30
        idx = _sample_indices(weights, k, rng)
        sampled = pts[idx]

        # Source points: small, muted, but fully opaque with edge
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=source_color,
                   alpha=0.7, zorder=1, edgecolors="white", linewidths=0.2,
                   label="source" if panel_idx == 1 else None)
        # Sampled points: larger, vivid, with dark edge
        ax.scatter(sampled[:, 0], sampled[:, 1], s=50, color=sample_color,
                   alpha=0.9, zorder=3, edgecolors="black", linewidths=0.5,
                   label="sampled" if panel_idx == 1 else None)
        # Vantage: largest, distinct shape
        ax.scatter(*vantage, s=180, color=vantage_color, marker="*",
                   zorder=5, edgecolors=vantage_edge, linewidths=0.8,
                   label="vantage" if panel_idx == 1 else None)
        ax.set_title(f"Sample {panel_idx}  ($k={k}$)", fontsize=10)

    for ax in axes:
        ax.set_xlim(-2.5, 5.5)
        ax.set_ylim(-2.5, 5.0)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$", fontsize=9)
        ax.set_ylabel("$x_2$", fontsize=9)
        ax.tick_params(labelsize=8)

    # Shared legend below the two sample panels
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.62, -0.01), ncol=3, fontsize=9,
               frameon=False, handletextpad=0.4, columnspacing=1.5)
    fig.subplots_adjust(bottom=0.17)

    return fig


def plot_sampling_gaussian(**kw):
    return _plot_sampling_row(
        "gaussian", {"mean": 0.0, "sigma": 1.5},
        r"Gaussian($\mu=0,\;\sigma=1.5$)", **kw)


def plot_sampling_uniform(**kw):
    return _plot_sampling_row(
        "uniform", {"lo": 2.0, "hi": 4.0},
        r"Uniform($[2,\;4]$)", **kw)


def plot_sampling_mixture(**kw):
    return _plot_sampling_row(
        "mixture", {
            "components": [
                {"type": "gaussian", "params": {"mean": 0.0, "sigma": 0.8}},
                {"type": "uniform", "params": {"lo": 3.0, "hi": 5.0}},
            ],
            "weights": [0.6, 0.4],
        },
        r"$0.6\;\mathrm{Gaussian}(\sigma\!=\!0.8) + 0.4\;\mathrm{Uniform}([3,5])$",
        **kw)


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

# Light theme: black vantage with white edge, dark edges on points
LIGHT_KW = {}  # all defaults

# Dark theme: white vantage with black edge, light edges, inverted cmap
DARK_KW = dict(
    source_color=OI_SKY_BLUE,
    sample_color=OI_ORANGE,
    vantage_color="white",
    vantage_edge="black",
    zero_color="#555555",
    cmap_name="viridis",
    cmap_range=(0.15, 1.0),
)

if __name__ == "__main__":
    gallery = [
        ("gallery_sampling_gaussian", lambda: plot_sampling_gaussian(**LIGHT_KW)),
        ("gallery_sampling_uniform", lambda: plot_sampling_uniform(**LIGHT_KW)),
        ("gallery_sampling_mixture", lambda: plot_sampling_mixture(**LIGHT_KW)),
    ]
    gallery_dark = [
        ("gallery_sampling_gaussian", lambda: plot_sampling_gaussian(**DARK_KW)),
        ("gallery_sampling_uniform", lambda: plot_sampling_uniform(**DARK_KW)),
        ("gallery_sampling_mixture", lambda: plot_sampling_mixture(**DARK_KW)),
    ]
    for (name, func), (_, func_dark) in zip(gallery, gallery_dark):
        _save_themed(func, *LIGHT, HERE / f"{name}_light.png")
        _save_themed(func_dark, *DARK, HERE / f"{name}_dark.png")
