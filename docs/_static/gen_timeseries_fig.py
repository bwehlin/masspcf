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

if __name__ == "__main__":
    for name, plot_func in [("timeseries_basic", plot_timeseries_basic),
                             ("timeseries_eval", plot_timeseries_eval),
                             ("timeseries_tensor", plot_timeseries_tensor),
                             ("timeseries_different_scales", plot_different_scales),
                             ("timeseries_datetime", plot_datetime_example)]:
        _save_themed(plot_func, *LIGHT, HERE / f"{name}_light.png")
        _save_themed(plot_func, *DARK, HERE / f"{name}_dark.png")
