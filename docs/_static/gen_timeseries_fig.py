"""Generate the time series example figures for the docs."""

import numpy as np
import masspcf as mpcf
from masspcf.plotting import plot as plotpcf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent

# -- Example time series used in the docs --
ts_sensor = mpcf.TimeSeries(
    np.array([22.1, 22.3, 23.0, 22.8, 22.5, 23.2, 24.0, 23.5]),
    start_time=0.0, time_step=1.0,
)

ts_short = mpcf.TimeSeries(
    np.array([18.0, 19.5, 20.0, 19.0]),
    start_time=2.0, time_step=1.0,
)


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_basic():
    ts = mpcf.TimeSeries(
        np.array([1.0, 3.0, 2.0, 4.0, 1.5]),
        start_time=10.0, time_step=2.0,
    )
    fig, ax = plt.subplots(figsize=(6, 2.5))
    times = ts.times
    values = ts.values
    # Draw horizontal segments
    for i in range(len(times)):
        t_start = times[i]
        t_end = times[i + 1] if i + 1 < len(times) else times[i] + (times[1] - times[0])
        ax.plot([t_start, t_end], [values[i], values[i]], linewidth=2, color="steelblue")
        if i + 1 < len(times):
            ax.plot(t_end, values[i], "o", markersize=4, color="steelblue",
                    markerfacecolor="white", markeredgewidth=1.5)
    ax.plot(times, values, "o", markersize=4, color="steelblue")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("TimeSeries(start_time=10.0, time_step=2.0)")
    ax.set_ylim(0, 5)
    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_eval():
    ts = mpcf.TimeSeries(
        np.array([1.0, 3.0, 2.0, 4.0, 1.5]),
        start_time=10.0, time_step=2.0,
    )
    fig, ax = plt.subplots(figsize=(6, 2.5))
    times = ts.times
    values = ts.values
    for i in range(len(times)):
        t_start = times[i]
        t_end = times[i + 1] if i + 1 < len(times) else times[i] + (times[1] - times[0])
        ax.plot([t_start, t_end], [values[i], values[i]], linewidth=2, color="steelblue")
        if i + 1 < len(times):
            ax.plot(t_end, values[i], "o", markersize=4, color="steelblue",
                    markerfacecolor="white", markeredgewidth=1.5)
    ax.plot(times, values, "o", markersize=4, color="steelblue")

    # Evaluation markers
    query_times = [13.0, 15.0]
    for t in query_times:
        val = ts(t)
        ax.axvline(t, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.plot(t, val, "ro", markersize=6, zorder=5)
        ax.annotate(f"ts({t}) = {val}", (t, val), textcoords="offset points",
                    xytext=(5, 8), fontsize=9)

    # NaN regions
    ax.axvspan(8, 10, alpha=0.3, color="gray")
    ax.axvspan(18, 20, alpha=0.3, color="gray")
    ax.text(9, 4.5, "NaN", ha="center", fontsize=9, fontstyle="italic", alpha=0.7)
    ax.text(19, 4.5, "NaN", ha="center", fontsize=9, fontstyle="italic", alpha=0.7)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Evaluation with out-of-range NaN")
    ax.set_xlim(8, 20)
    ax.set_ylim(0, 5)
    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_tensor():
    ts1 = mpcf.TimeSeries(
        np.array([1.0, 3.0, 2.0, 4.0]),
        start_time=0.0, time_step=1.0,
    )
    ts2 = mpcf.TimeSeries(
        np.array([2.0, 1.0, 3.0]),
        start_time=1.0, time_step=1.5,
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5), sharey=True)

    for ax, ts, label in [(ax1, ts1, "ts1 (start_time=0, step=1)"),
                           (ax2, ts2, "ts2 (start_time=1, step=1.5)")]:
        times = ts.times
        values = ts.values
        for i in range(len(times)):
            t_start = times[i]
            t_end = times[i + 1] if i + 1 < len(times) else times[i] + (times[1] - times[0])
            ax.plot([t_start, t_end], [values[i], values[i]], linewidth=2, color="steelblue")
            if i + 1 < len(times):
                ax.plot(t_end, values[i], "o", markersize=4, color="steelblue",
                        markerfacecolor="white", markeredgewidth=1.5)
        ax.plot(times, values, "o", markersize=4, color="steelblue")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Time")

    ax1.set_ylabel("Value")
    fig.suptitle("TimeSeriesTensor([ts1, ts2])", fontsize=11)
    fig.tight_layout()
    return fig


def _plot_ts_segments(ax, ts, color="steelblue"):
    """Helper: draw a piecewise-constant time series on an axes."""
    times = ts.times
    values = ts.values
    step = times[1] - times[0] if len(times) > 1 else 1
    for i in range(len(times)):
        t_start = times[i]
        t_end = times[i + 1] if i + 1 < len(times) else times[i] + step
        ax.plot([t_start, t_end], [values[i], values[i]],
                linewidth=2, color=color)
        if i + 1 < len(times):
            ax.plot(t_end, values[i], "o", markersize=4, color=color,
                    markerfacecolor="white", markeredgewidth=1.5)
    ax.plot(times, values, "o", markersize=4, color=color)


# -- included in docs via literalinclude :pyobject: --
def plot_different_scales():
    # Two sensors: one fast, one slow, starting at different times
    fast = mpcf.TimeSeries(
        np.array([2.1, 2.5, 3.0, 2.8, 2.3, 2.9, 3.2, 2.7, 2.4, 3.1]),
        start_time=1.0, time_step=0.5,
    )
    slow = mpcf.TimeSeries(
        np.array([10.0, 12.0, 11.5, 13.0, 12.5]),
        start_time=0.0, time_step=1.5,
    )
    tensor = mpcf.TimeSeriesTensor([fast, slow])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)

    _plot_ts_segments(ax1, fast, color="steelblue")
    ax1.set_ylabel("Temperature")
    ax1.set_title("Sensor A: start=1.0, step=0.5s (fast)", fontsize=10)

    _plot_ts_segments(ax2, slow, color="#e67e22")
    ax2.set_ylabel("Pressure")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Sensor B: start=0.0, step=1.5s (slow)", fontsize=10)

    # Mark a shared query time
    t_query = 3.5
    for ax, ts, color in [(ax1, fast, "steelblue"), (ax2, slow, "#e67e22")]:
        val = ts(t_query)
        ax.axvline(t_query, color="red", linestyle="--", linewidth=1, alpha=0.7)
        if not np.isnan(val):
            ax.plot(t_query, val, "ro", markersize=6, zorder=5)
            ax.annotate(f"{val:.1f}", (t_query, val),
                        textcoords="offset points", xytext=(6, 5), fontsize=9)

    fig.suptitle(f"tensor({t_query}) evaluates both at the same real time",
                 fontsize=11)
    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_datetime_example():
    epoch1 = np.datetime64("2024-06-15T08:00:00")
    epoch2 = np.datetime64("2024-06-15T08:00:02")

    ts1 = mpcf.TimeSeries(
        np.array([22.1, 22.3, 23.0, 22.8, 22.5, 23.2, 24.0, 23.5]),
        start_time=epoch1, time_step=np.timedelta64(500, "ms"),
    )
    ts2 = mpcf.TimeSeries(
        np.array([21.0, 21.8, 22.5, 23.1, 22.9]),
        start_time=epoch2, time_step=np.timedelta64(1, "s"),
    )
    tensor = mpcf.TimeSeriesTensor([ts1, ts2])

    # Evaluate both series on a dense grid of datetime query times
    query_times = np.arange(
        np.datetime64("2024-06-15T07:59:59.500"),
        np.datetime64("2024-06-15T08:00:07.500"),
        np.timedelta64(50, "ms"),
    )
    # Evaluate each series via __call__ on the individual TimeSeries;
    # result shape: (n_times,) per series
    vals1 = ts1(query_times)
    vals2 = ts2(query_times)

    # Seconds relative to 08:00:00 for a readable x-axis
    ref = epoch1
    t_sec = (query_times - ref) / np.timedelta64(1, "ms") / 1000

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    colors = ["steelblue", "#e67e22"]

    for ax, vals, color, title in [
        (ax1, vals1, colors[0], "Sensor A: start 08:00:00, step 500ms"),
        (ax2, vals2, colors[1], "Sensor B: start 08:00:02, step 1s"),
    ]:
        mask = ~np.isnan(vals)
        ax.step(t_sec[mask], vals[mask], where="post", linewidth=2, color=color)
        ax.set_ylabel("Temp (C)")
        ax.set_title(title, fontsize=10)
        # Pad y-axis so annotations at extremes aren't clipped
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - 0.3, ymax + 0.3)

    ax2.set_xlabel("Seconds after 08:00:00")

    # Mark a query time -- evaluate both at once via the tensor
    t_q = np.datetime64("2024-06-15T08:00:02.700")
    t_q_sec = (t_q - ref) / np.timedelta64(1, "ms") / 1000
    result = tensor(t_q)  # array([23.2, 21.0])
    for ax, val in [(ax1, result[0]), (ax2, result[1])]:
        ax.axvline(t_q_sec, color="red", linestyle="--", linewidth=1, alpha=0.7)
        if not np.isnan(val):
            ax.plot(t_q_sec, val, "ro", markersize=6, zorder=5)
            ax.annotate(f"{val:.1f}", (t_q_sec, val),
                        textcoords="offset points", xytext=(8, 10), fontsize=9,
                        zorder=10, bbox=dict(boxstyle="round,pad=0.15",
                                             fc=ax.get_facecolor(), ec="none",
                                             alpha=0.8))

    fig.suptitle("Datetime time series with different start times and rates",
                 fontsize=11)
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

# -- included in docs via literalinclude :pyobject: --
def plot_multichannel():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([
        [22.1, 45.0],
        [22.5, 44.0],
        [23.0, 43.5],
        [23.8, 42.0],
        [23.2, 43.0],
    ])
    ts = mpcf.TimeSeries(times, values)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)

    _plot_ts_segments(ax1, mpcf.TimeSeries(times, values[:, 0]),
                      color="steelblue")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Channel 0: Temperature", fontsize=10)

    _plot_ts_segments(ax2, mpcf.TimeSeries(times, values[:, 1]),
                      color="#e67e22")
    ax2.set_ylabel("Humidity (%)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Channel 1: Humidity", fontsize=10)

    # Mark evaluation at t=1.5
    t_q = 1.5
    result = ts(t_q)
    for ax, val, color in [(ax1, result[0], "steelblue"),
                            (ax2, result[1], "#e67e22")]:
        ax.axvline(t_q, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.plot(t_q, val, "ro", markersize=6, zorder=5)
        ax.annotate(f"{val:.1f}", (t_q, val),
                    textcoords="offset points", xytext=(6, 5), fontsize=9)

    fig.suptitle(f"Multi-channel TimeSeries: ts({t_q}) = "
                 f"[{result[0]:.1f}, {result[1]:.1f}]", fontsize=11)
    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_embed_basic():
    # A simple signal: noisy sinusoid
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 200)
    values = np.sin(t) + 0.1 * np.random.randn(len(t))
    ts = mpcf.TimeSeries(values, start_time=0.0, time_step=t[1] - t[0])

    cloud = mpcf.embed_time_delay(ts, dimension=2, delay=0.4)
    pts = np.asarray(cloud[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # Left: original time series
    ax1.step(t, values, where="post", linewidth=1.5, color="steelblue")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("x(t)")
    ax1.set_title("Original time series", fontsize=10)

    # Right: 2D embedding
    ax2.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.6, color="steelblue")
    ax2.set_xlabel(r"$x(t - \tau)$")
    ax2.set_ylabel(r"$x(t)$")
    ax2.set_title(r"Time delay embedding ($d=2, \tau=0.4$)", fontsize=10)
    ax2.set_aspect("equal")

    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_embed_windowed():
    # Longer signal with distinct phases
    np.random.seed(7)
    n = 300
    t = np.linspace(0, 6 * np.pi, n)
    values = np.sin(t) + 0.5 * np.sin(3 * t) + 0.08 * np.random.randn(n)
    ts = mpcf.TimeSeries(values, start_time=0.0, time_step=t[1] - t[0])

    clouds = mpcf.embed_time_delay(
        ts, dimension=2, delay=0.3, window=4.0, stride=4.0)

    n_win = clouds.shape[0]
    fig, axes = plt.subplots(1, min(n_win, 4), figsize=(12, 3),
                              sharex=True, sharey=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]
    colors = ["steelblue", "#e67e22", "#27ae60", "#8e44ad"]
    for i, ax in enumerate(axes[:n_win]):
        pts = np.asarray(clouds[i])
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.6, color=colors[i % 4])
        ax.set_title(f"Window {i}", fontsize=10)
        ax.set_xlabel(r"$x(t-\tau)$")
        if i == 0:
            ax.set_ylabel(r"$x(t)$")
        ax.set_aspect("equal")
    fig.suptitle(r"Windowed embedding ($d=2, \tau=0.3$, window=4.0)",
                 fontsize=11)
    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_interpolation():
    values = np.array([1.0, 3.0, 2.0, 4.0, 1.5])
    ts_nearest = mpcf.TimeSeries(values, start_time=10.0, time_step=2.0)
    ts_linear = mpcf.TimeSeries(values, start_time=10.0, time_step=2.0,
                                interpolation='linear')

    t_dense = np.linspace(10.0, 18.0, 200)
    y_nearest = ts_nearest(t_dense)
    y_linear = ts_linear(t_dense)
    times = ts_nearest.times

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5),
                                    sharey=True)

    # Left: nearest (step function)
    ax1.step(t_dense, y_nearest, where="post", linewidth=2,
             color="steelblue")
    ax1.plot(times, values, "o", markersize=5, color="steelblue")
    ax1.set_title("interpolation='nearest'", fontsize=10)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")

    # Right: linear
    ax2.plot(t_dense, y_linear, linewidth=2, color="#e67e22")
    ax2.plot(times, values, "o", markersize=5, color="#e67e22")
    ax2.set_title("interpolation='linear'", fontsize=10)
    ax2.set_xlabel("Time")

    fig.suptitle("Interpolation modes", fontsize=11)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    for name, plot_func in [("timeseries_basic", plot_timeseries_basic),
                             ("timeseries_eval", plot_timeseries_eval),
                             ("timeseries_tensor", plot_timeseries_tensor),
                             ("timeseries_different_scales", plot_different_scales),
                             ("timeseries_datetime", plot_datetime_example),
                             ("timeseries_multichannel", plot_multichannel),
                             ("timeseries_embed_basic", plot_embed_basic),
                             ("timeseries_embed_windowed", plot_embed_windowed),
                             ("timeseries_interpolation", plot_interpolation)]:
        _save_themed(plot_func, *LIGHT, HERE / f"{name}_light.png")
        _save_themed(plot_func, *DARK, HERE / f"{name}_dark.png")
