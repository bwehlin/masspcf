"""Hybrid persistence benchmark harness.

Five sweeps, one subcommand each, written to benchmarks/bench_results/*.csv:

    crossover       Q1: per-item CPU vs GPU as a function of n.
                        Identifies the n below which GPU dispatch
                        is wasted (kernel/transfer overhead dominates).

    cpucap          Q2: per-batch wall as CPU worker count varies,
                        for several item sizes. Identifies the n above
                        which any free CPU core is more productive
                        spent waiting on the GPU than running its own
                        item.

    cooperate       Q3: batch wall with fixed n, varying GPU
                        concurrency cap M and CPU worker count.
                        Identifies the (n, M) regime where strict
                        cooperation beats both gpu-only and cpu-only.

    distribution    Q4: per-item CPU vs GPU at fixed n, varying point
                        cloud distribution (uniform, gaussian,
                        circle, sphere, torus, two clusters). Tests
                        whether the crossover heuristic is
                        distribution-sensitive.

    dimensionality  Q5: per-item CPU vs GPU at fixed n, varying
                        ambient dimension. Tests whether the
                        crossover heuristic depends on d.

    all             Run every sweep above in order.

Each timing measurement is a fresh `compute_persistent_homology` call,
preceded by one untimed warm-up; the timed runs are reported as median
of 3 (or 1 for n >= 2000, where each run is already long).

The harness uses `mpcf.system.limit_cpus`, `limit_gpu_concurrency`, and
`force_cpu` as runtime knobs; it does not import any private internals.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil

import masspcf as mpcf
import masspcf.persistence as mpers
import masspcf.system as mpsys
from masspcf import _mpcf_cpp as cpp
from masspcf.persistence.ripser import _ripser_plusplus_available

try:
    import nvtx as _nvtx
except ImportError:
    _nvtx = None


@contextmanager
def _nvtx_range(name: str):
    """Optional NVTX range. No-op if the nvtx package is not installed.
    Used to mark the timed compute_persistent_homology call so that
    nsys stats can filter its CUDA-API totals to the timed window."""
    if _nvtx is None:
        yield
        return
    with _nvtx.annotate(name):
        yield


def _scheduler_stats_supported() -> bool:
    return hasattr(cpp, "get_last_gpu_scheduler_stats")


def _reset_scheduler_stats() -> None:
    if hasattr(cpp, "reset_last_gpu_scheduler_stats"):
        cpp.reset_last_gpu_scheduler_stats()


def _read_scheduler_stats() -> dict:
    if _scheduler_stats_supported():
        return dict(cpp.get_last_gpu_scheduler_stats())
    return {}


RESULTS_DIR = Path(__file__).parent / "bench_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GPU_AVAILABLE = _ripser_plusplus_available()


# ---------------------------------------------------------------------------
# Synthetic point clouds
# ---------------------------------------------------------------------------


def gen_uniform(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random((n, d), dtype=np.float64)


def gen_gaussian(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((n, d))


def gen_circle(n: int, rng: np.random.Generator) -> np.ndarray:
    t = rng.random(n) * 2 * np.pi
    X = np.stack([np.cos(t), np.sin(t)], axis=1)
    X += 0.01 * rng.standard_normal(X.shape)
    return X


def gen_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.standard_normal((n, 3))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def gen_torus(n: int, rng: np.random.Generator, R: float = 2.0, r: float = 0.5) -> np.ndarray:
    u = rng.random(n) * 2 * np.pi
    v = rng.random(n) * 2 * np.pi
    X = np.stack([
        (R + r * np.cos(v)) * np.cos(u),
        (R + r * np.cos(v)) * np.sin(u),
        r * np.sin(v),
    ], axis=1)
    return X


def gen_two_clusters(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    half = n // 2
    a = rng.standard_normal((half, d))
    b = rng.standard_normal((n - half, d)) + np.array([5.0] + [0.0] * (d - 1))
    return np.vstack([a, b])


DISTRIBUTIONS: dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    "uniform_3d": lambda n, rng: gen_uniform(n, 3, rng),
    "gaussian_3d": lambda n, rng: gen_gaussian(n, 3, rng),
    "circle": gen_circle,
    "sphere": gen_sphere,
    "torus": gen_torus,
    "two_clusters_3d": lambda n, rng: gen_two_clusters(n, 3, rng),
}


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


@dataclass
class Timing:
    median_s: float
    min_s: float
    runs: list[float]


# ---------------------------------------------------------------------------
# CPU + GPU resource monitor
# ---------------------------------------------------------------------------


@dataclass
class MonitorSample:
    t_rel_s: float
    cpu_total_pct: float
    cpu_per_core: list[float]
    ram_used_mb: float
    gpu_mem_mb: list[float]   # one entry per visible GPU
    gpu_util_pct: list[float] # one entry per visible GPU


class SystemMonitor:
    """Background sampler for CPU (psutil) and GPU (nvidia-smi) usage.

    Use as a context manager around a timed call:

        with SystemMonitor() as mon:
            run_workload()
        for s in mon.samples: ...

    Sampling cadence is `interval_s`. nvidia-smi is invoked once per tick
    via subprocess; if it is missing, GPU columns stay empty. CPU sampling
    uses psutil (cheap, in-process).
    """

    def __init__(self, interval_s: float = 0.1):
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None
        self._n_gpus = self._query_n_gpus() if self._has_nvidia_smi else 0
        self.samples: list[MonitorSample] = []

    def _query_n_gpus(self) -> int:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=2.0, check=True,
            )
            return sum(1 for line in out.stdout.splitlines() if line.strip())
        except (subprocess.SubprocessError, FileNotFoundError):
            return 0

    def _sample_gpus(self) -> tuple[list[float], list[float]]:
        if self._n_gpus == 0:
            return [], []
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2.0, check=True,
            )
            mems, utils = [], []
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 2:
                    continue
                mems.append(float(parts[0]))
                utils.append(float(parts[1]))
            return mems, utils
        except (subprocess.SubprocessError, ValueError):
            return [0.0] * self._n_gpus, [0.0] * self._n_gpus

    def _loop(self):
        # Prime psutil so the first measurement is non-zero.
        psutil.cpu_percent(interval=None, percpu=True)
        psutil.cpu_percent(interval=None)
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            t = time.perf_counter()
            cpu_total = psutil.cpu_percent(interval=None)
            cpu_each = psutil.cpu_percent(interval=None, percpu=True)
            ram_mb = psutil.virtual_memory().used / 1e6
            gpu_mem, gpu_util = self._sample_gpus()
            self.samples.append(MonitorSample(
                t_rel_s=t - self._t0,
                cpu_total_pct=cpu_total,
                cpu_per_core=cpu_each,
                ram_used_mb=ram_mb,
                gpu_mem_mb=gpu_mem,
                gpu_util_pct=gpu_util,
            ))
            next_tick += self._interval
            sleep_for = max(0.0, next_tick - time.perf_counter())
            if self._stop.wait(sleep_for):
                break

    def __enter__(self) -> "SystemMonitor":
        self._t0 = time.perf_counter()
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def n_gpus(self) -> int:
        return self._n_gpus

    def peak_gpu_mem_mb(self) -> list[float]:
        if not self.samples or self._n_gpus == 0:
            return []
        return [max((s.gpu_mem_mb[i] for s in self.samples if i < len(s.gpu_mem_mb)), default=0.0)
                for i in range(self._n_gpus)]

    def peak_gpu_util_pct(self) -> list[float]:
        if not self.samples or self._n_gpus == 0:
            return []
        return [max((s.gpu_util_pct[i] for s in self.samples if i < len(s.gpu_util_pct)), default=0.0)
                for i in range(self._n_gpus)]

    def avg_gpu_util_pct(self) -> list[float]:
        if not self.samples or self._n_gpus == 0:
            return []
        out = []
        for i in range(self._n_gpus):
            vals = [s.gpu_util_pct[i] for s in self.samples if i < len(s.gpu_util_pct)]
            out.append(statistics.mean(vals) if vals else 0.0)
        return out

    def avg_cpu_pct(self) -> float:
        if not self.samples:
            return 0.0
        return statistics.mean(s.cpu_total_pct for s in self.samples)


def _time_call(fn: Callable[[], None], repeats: int) -> Timing:
    fn()  # warm-up (untimed)
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - t0)
    return Timing(median_s=statistics.median(runs), min_s=min(runs), runs=runs)


def time_per_item(X: np.ndarray, device: str, max_dim: int = 1, repeats: int | None = None) -> Timing:
    """Time one persistent-homology call on a single point cloud."""
    if repeats is None:
        # max_dim >= 2 work grows super-linearly; trim repeats more aggressively.
        large = X.shape[0] >= 2000 or (max_dim >= 2 and X.shape[0] >= 500)
        repeats = 1 if large else 3

    def call():
        mpers.compute_persistent_homology(X, max_dim=max_dim, device=device)

    return _time_call(call, repeats)


@dataclass
class BatchResult:
    wall_s: float
    monitor: SystemMonitor | None = None
    sched_stats: dict = field(default_factory=dict)


def time_batch(items: list[np.ndarray], device: str, max_dim: int = 1,
               monitor: bool = False, **_: object) -> BatchResult:
    """Wrap items into a PointCloudTensor and time one batched call.

    When `monitor=True`, captures a CPU+GPU resource timeseries via
    `SystemMonitor` for the duration of the timed call (warm-up excluded).
    Also captures the post-run scheduler stats snapshot when GPU is
    involved (problem-agnostic counters: admitted / failed / oom /
    peak_active). The snapshot reflects only the timed call -- it is
    reset just before the call so warm-up activity is excluded.
    """
    n_items = len(items)
    X = mpcf.zeros((n_items,), dtype=mpcf.pcloud64)
    for i, pc in enumerate(items):
        X[i] = pc.astype(np.float64, copy=False)

    # Warm-up the very first dispatch path of this device.
    mpers.compute_persistent_homology(X[:1], max_dim=max_dim, device=device)
    _reset_scheduler_stats()

    mon: SystemMonitor | None = None
    if monitor:
        mon = SystemMonitor()
        mon.__enter__()
    try:
        t0 = time.perf_counter()
        # Mark the timed PH call with an NVTX range so nsys stats can
        # scope CUDA-API totals to just this window (the bench also
        # runs a cpu_only pre-pass + CUDA context init + matplotlib,
        # which otherwise inflate the global totals). If nvtx is not
        # installed the range becomes a no-op.
        with _nvtx_range("ph_compute"):
            mpers.compute_persistent_homology(X, max_dim=max_dim, device=device)
        wall = time.perf_counter() - t0
    finally:
        if mon is not None:
            mon.__exit__(None, None, None)
    return BatchResult(wall_s=wall, monitor=mon, sched_stats=_read_scheduler_stats())


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> Path:
    path = RESULTS_DIR / f"{name}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save_fig(name: str) -> Path:
    path = RESULTS_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    render_index_html()
    return path


_SWEEP_INFO: dict[str, tuple[str, str]] = {
    "crossover": (
        "Q1 — Per-item CPU vs GPU vs n",
        "Where does GPU dispatch start to pay off? Below the crossover, "
        "kernel-launch and transfer overhead dominate and the item should "
        "stay on the CPU.",
    ),
    "cpucap": (
        "Q2 — Batch wall vs CPU worker count",
        "How does CPU-only batch wall scale with worker count, per item "
        "size? Saturation here tells us when freeing a CPU core to wait "
        "on the GPU is the better trade.",
    ),
    "cooperate": (
        "Q3 — Hybrid wall vs GPU concurrency cap",
        "For a fixed (n, batch), how does total wall change as we vary the "
        "max number of concurrent GPU jobs? The dashed reference is the "
        "CPU-only baseline.",
    ),
    "cooperate_timeline": (
        "Q3 timeline — CPU and GPU usage during each cap",
        "Per-cap resource timeline: CPU%% (left axis, red) and per-GPU "
        "memory used (right axis). Sampled every 100 ms via psutil and "
        "nvidia-smi.",
    ),
    "distribution": (
        "Q4 — Per-item time across point cloud distributions",
        "At fixed n, does the geometry of the point cloud change the "
        "CPU/GPU balance? Reduction-matrix density depends on the "
        "filtration structure.",
    ),
    "dimensionality": (
        "Q5 — Per-item time vs ambient dimension",
        "At fixed n, does the ambient dimension d shift the crossover? "
        "Distance computation is O(n^2 * d) but reduction is independent "
        "of d.",
    ),
}


def render_index_html() -> Path:
    """Rebuild bench_results/index.html with whichever sweeps exist."""
    sections = []
    for name, (title, desc) in _SWEEP_INFO.items():
        png = RESULTS_DIR / f"{name}.png"
        csv_path = RESULTS_DIR / f"{name}.csv"
        if not png.exists():
            continue
        csv_html = ""
        if csv_path.exists():
            csv_text = csv_path.read_text()
            csv_html = (
                "<details><summary>Show CSV</summary>"
                f"<pre>{csv_text}</pre></details>"
            )
        sections.append(
            f"<section>"
            f"<h2>{title}</h2>"
            f"<p>{desc}</p>"
            f'<img src="{png.name}" alt="{title}">'
            f"{csv_html}"
            f"</section>"
        )

    if not sections:
        body = "<p><em>No sweeps have been run yet.</em></p>"
    else:
        body = "\n".join(sections)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hybrid PH benchmark results</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 920px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1 {{ margin-bottom: 0.2em; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 2em; }}
  section {{ margin-bottom: 2.5em; }}
  section h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
  img {{ max-width: 100%; border: 1px solid #eee; }}
  pre {{ background: #f4f4f4; padding: 0.6em; overflow: auto; font-size: 0.85em; }}
  details {{ margin-top: 0.6em; }}
  summary {{ cursor: pointer; color: #336; }}
</style>
</head>
<body>
<h1>Hybrid persistence benchmark</h1>
<div class="meta">Generated {timestamp}. Sweeps live in <code>benchmarks/bench_results/</code>.</div>
{body}
</body>
</html>
"""
    path = RESULTS_DIR / "index.html"
    path.write_text(html)
    return path


def plot_crossover(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    by_dev: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        by_dev.setdefault(r["device"], []).append((int(r["n"]), float(r["median_s"])))
    for dev, pts in by_dev.items():
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=dev.upper())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("point cloud size n")
    ax.set_ylabel("per-item time [s]")
    ax.set_title("Q1: per-item PH time, CPU vs GPU (gaussian_3d)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return _save_fig("crossover")


def plot_cpucap(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    by_n: dict[int, list[tuple[int, float]]] = {}
    for r in rows:
        by_n.setdefault(int(r["n"]), []).append((int(r["cpu_cap"]), float(r["wall_s"])))
    for n, pts in sorted(by_n.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=f"n={n}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CPU worker cap")
    ax.set_ylabel("batch wall time [s]")
    ax.set_title("Q2: batch wall vs CPU worker count (gaussian_3d, CPU only)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return _save_fig("cpucap")


def plot_cooperate(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    cpu_only_wall = next((float(r["wall_s"]) for r in rows if r["mode"] == "cpu_only"), None)
    hybrid_rows = [r for r in rows if r["mode"] == "hybrid"]
    hybrid_rows.sort(key=lambda r: int(r["gpu_cap"]) if r["gpu_cap"] != "-" else 0)
    labels = [("uncapped" if int(r["gpu_cap"]) == 0 else f"M={r['gpu_cap']}") for r in hybrid_rows]
    walls = [float(r["wall_s"]) for r in hybrid_rows]
    xs = list(range(len(hybrid_rows)))
    bars = ax.bar(xs, walls, color="tab:blue", label="hybrid (GPU+CPU)")
    if cpu_only_wall is not None:
        ax.axhline(cpu_only_wall, color="tab:red", linestyle="--", label="cpu_only")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("batch wall time [s]")
    n = next((int(r["n"]) for r in rows), "?")
    M = next((int(r["batch"]) for r in rows), "?")
    ax.set_title(f"Q3: cooperate sweep (n={n}, batch={M})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    # Annotate each bar with the scheduler dispatch breakdown.
    for bar, r in zip(bars, hybrid_rows):
        admit = r.get("gpu_admitted", "-")
        room = r.get("cpu_fallback_no_room", "-")
        cap = r.get("cpu_fallback_cap", "-")
        oom = r.get("oom", "-")
        peak = r.get("peak_active", "-")
        label = f"GPU={admit}\nCPU(room)={room}\nCPU(cap)={cap}\nOOM={oom}\npeak={peak}"
        ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="#333")
    return _save_fig("cooperate")


def plot_cooperate_timeline(runs: list[tuple[str, "BatchResult"]]) -> Path:
    """Stacked timeseries of CPU %, GPU util %, GPU memory MB across cooperate caps."""
    runs = [(label, br) for label, br in runs if br.monitor is not None and br.monitor.samples]
    if not runs:
        return RESULTS_DIR / "cooperate_timeline.png"

    n_gpus = max(br.monitor.n_gpus() for _, br in runs)
    fig, axes = plt.subplots(len(runs), 1, figsize=(9, 2.8 * len(runs)),
                             sharex=False, squeeze=False)
    for ax, (label, br) in zip(axes[:, 0], runs):
        ts = [s.t_rel_s for s in br.monitor.samples]

        # Left axis: percentages (CPU total, per-GPU util).
        ax.plot(ts, [s.cpu_total_pct for s in br.monitor.samples],
                color="tab:red", label="CPU %", linewidth=1.2)
        for g in range(n_gpus):
            utils = [s.gpu_util_pct[g] for s in br.monitor.samples if g < len(s.gpu_util_pct)]
            if utils:
                ax.plot(ts[:len(utils)], utils, color="tab:orange",
                        label=f"GPU{g} util %", linewidth=1.2, linestyle="--")
        ax.set_ylim(0, 105)
        ax.set_ylabel("percent")
        ax.legend(loc="upper left", fontsize=8)

        # Right axis: GPU memory in MB.
        if n_gpus > 0:
            ax2 = ax.twinx()
            for g in range(n_gpus):
                mems = [s.gpu_mem_mb[g] for s in br.monitor.samples if g < len(s.gpu_mem_mb)]
                if mems:
                    ax2.plot(ts[:len(mems)], mems, color="tab:blue",
                             label=f"GPU{g} mem MB", linewidth=1.2)
            ax2.set_ylabel("GPU mem [MB]", color="tab:blue")
            ax2.tick_params(axis="y", labelcolor="tab:blue")

        ax.set_title(f"{label}  (wall={br.wall_s:.2f}s)", fontsize=10, loc="left")
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_xlabel("time [s]")
    fig.suptitle("Q3 timeline: CPU % / GPU util % (left) and GPU mem MB (right)")
    return _save_fig("cooperate_timeline")


def plot_distribution(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dists = sorted({r["distribution"] for r in rows})
    devices = sorted({r["device"] for r in rows})
    width = 0.8 / len(devices)
    xs = np.arange(len(dists))
    for i, dev in enumerate(devices):
        ys = []
        for d in dists:
            r = next((r for r in rows if r["distribution"] == d and r["device"] == dev), None)
            ys.append(float(r["median_s"]) if r else 0.0)
        ax.bar(xs + i * width, ys, width=width, label=dev.upper())
    ax.set_xticks(xs + (len(devices) - 1) * width / 2)
    ax.set_xticklabels(dists, rotation=20, ha="right")
    ax.set_ylabel("per-item time [s]")
    n = next((int(r["n"]) for r in rows), "?")
    ax.set_title(f"Q4: per-item PH time across distributions (n={n})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    return _save_fig("distribution")


def plot_dimensionality(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    by_dev: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        by_dev.setdefault(r["device"], []).append((int(r["d"]), float(r["median_s"])))
    for dev, pts in by_dev.items():
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=dev.upper())
    ax.set_xlabel("ambient dimension d")
    ax.set_ylabel("per-item time [s]")
    ax.set_yscale("log")
    n = next((int(r["n"]) for r in rows), "?")
    ax.set_title(f"Q5: per-item PH time vs ambient dimension (n={n}, gaussian)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return _save_fig("dimensionality")


# ---------------------------------------------------------------------------
# Knob context managers
# ---------------------------------------------------------------------------


@contextmanager
def _cpu_cap(n: int | None):
    if n is None:
        yield
        return
    mpsys.limit_cpus(n)
    try:
        yield
    finally:
        # Best-effort reset to all cores.
        mpsys.limit_cpus(os.cpu_count() or 1)


@contextmanager
def _gpu_cap(n: int):
    mpsys.limit_gpu_concurrency(n)
    try:
        yield
    finally:
        mpsys.limit_gpu_concurrency(0)


@contextmanager
def _gpu_budget(f: float | None):
    if f is None:
        yield
        return
    mpsys.set_gpu_budget_fraction(f)
    try:
        yield
    finally:
        mpsys.set_gpu_budget_fraction(0.6)


@contextmanager
def _gpu_queue(on: bool):
    if not on or not hasattr(mpsys, "set_hybrid_gpu_queue_on_busy"):
        yield
        return
    mpsys.set_hybrid_gpu_queue_on_busy(True)
    try:
        yield
    finally:
        mpsys.set_hybrid_gpu_queue_on_busy(False)


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def sweep_crossover(args) -> None:
    ns = args.n_values or [50, 100, 250, 500, 1000, 2000, 5000]
    max_dim = args.max_dim
    rng = np.random.default_rng(args.seed)
    rows = []
    print(f"# crossover sweep: n in {ns}, distribution=gaussian_3d, max_dim={max_dim}", flush=True)
    for n in ns:
        X = gen_gaussian(n, 3, rng)
        for device in ("cpu", "gpu") if GPU_AVAILABLE else ("cpu",):
            t = time_per_item(X, device=device, max_dim=max_dim)
            rows.append(dict(
                sweep="crossover",
                distribution="gaussian_3d",
                n=n,
                d=3,
                max_dim=max_dim,
                device=device,
                median_s=f"{t.median_s:.6f}",
                min_s=f"{t.min_s:.6f}",
                runs=";".join(f"{x:.6f}" for x in t.runs),
            ))
            print(f"  n={n:>5d} max_dim={max_dim} {device:>3s} median={t.median_s:.4f}s min={t.min_s:.4f}s", flush=True)
    path = write_csv("crossover", list(rows[0].keys()), rows)
    plot_path = plot_crossover(rows)
    print(f"# wrote {path}", flush=True)
    print(f"# wrote {plot_path}", flush=True)


def sweep_cpucap(args) -> None:
    if not GPU_AVAILABLE:
        print("# cpucap: skipped (GPU not available)", flush=True)
        return
    rng = np.random.default_rng(args.seed)
    n_values = args.n_values or [500, 1000, 2000, 5000]
    cpu_caps = [1, 2, 4, 8, 16, os.cpu_count() or 1]
    cpu_caps = sorted(set(cpu_caps))
    M = args.batch_size
    max_dim = args.max_dim

    rows = []
    print(f"# cpucap sweep: n in {n_values}, batch={M}, cpu_caps={cpu_caps}, max_dim={max_dim}", flush=True)
    for n in n_values:
        items = [gen_gaussian(n, 3, rng) for _ in range(M)]
        for cap in cpu_caps:
            with _cpu_cap(cap):
                br = time_batch(items, device="cpu", max_dim=max_dim)
            rows.append(dict(
                sweep="cpucap",
                n=n,
                batch=M,
                max_dim=max_dim,
                cpu_cap=cap,
                wall_s=f"{br.wall_s:.6f}",
                per_item_s=f"{br.wall_s/M:.6f}",
            ))
            print(f"  n={n:>5d} M={M:>3d} max_dim={max_dim} cpu_cap={cap:>3d} "
                  f"wall={br.wall_s:.3f}s per_item={br.wall_s/M:.4f}s", flush=True)
    path = write_csv("cpucap", list(rows[0].keys()), rows)
    plot_path = plot_cpucap(rows)
    print(f"# wrote {path}", flush=True)
    print(f"# wrote {plot_path}", flush=True)


def sweep_cooperate(args) -> None:
    if not GPU_AVAILABLE:
        print("# cooperate: skipped (GPU not available)", flush=True)
        return
    rng = np.random.default_rng(args.seed)
    n = args.n
    M = args.batch_size
    max_dim = args.max_dim
    budget = getattr(args, "budget", None)
    queue_on_busy = getattr(args, "queue_on_busy", False)

    items = [gen_gaussian(n, 3, rng) for _ in range(M)]

    rows = []
    runs: list[tuple[str, BatchResult]] = []
    tag = f"n={n}, batch={M}, max_dim={max_dim}"
    if budget is not None:
        tag += f", budget={budget}"
    if queue_on_busy:
        tag += ", queue_on_busy"
    print(f"# cooperate sweep: {tag}", flush=True)

    def _row(mode, gpu_cap, br):
        s = br.sched_stats
        cap_fb = s.get("total_failed_cap", 0)
        room_fb = s.get("total_failed_no_room", 0)
        return dict(
            sweep="cooperate", n=n, batch=M, mode=mode, gpu_cap=gpu_cap,
            wall_s=f"{br.wall_s:.6f}", per_item_s=f"{br.wall_s/M:.6f}",
            avg_cpu_pct=f"{br.monitor.avg_cpu_pct():.1f}" if br.monitor else "-",
            peak_gpu_mem_mb=";".join(f"{x:.0f}" for x in br.monitor.peak_gpu_mem_mb()) if br.monitor else "",
            gpu_admitted=s.get("total_admitted", "-"),
            cpu_fallback_no_room=room_fb if s else "-",
            cpu_fallback_cap=cap_fb if s else "-",
            oom=s.get("total_oom", "-"),
            peak_active=s.get("peak_active", "-"),
        )

    def _summary(br):
        s = br.sched_stats
        if not s:
            return ""
        return (f" sched[admit={s['total_admitted']} "
                f"cpu_fb_room={s['total_failed_no_room']} "
                f"cpu_fb_cap={s['total_failed_cap']} "
                f"oom={s['total_oom']} peak_active={s['peak_active']}]")

    def _gpu_summary(br):
        if br.monitor is None:
            return ""
        mems = br.monitor.peak_gpu_mem_mb()
        peaks = br.monitor.peak_gpu_util_pct()
        avgs = br.monitor.avg_gpu_util_pct()
        if not mems:
            return ""
        mem_str = ",".join(f"{x:.0f}" for x in mems)
        peak_str = ",".join(f"{x:.0f}" for x in peaks)
        avg_str = ",".join(f"{x:.0f}" for x in avgs)
        return (f" peak_gpu_mem={mem_str}MB peak_util={peak_str}% avg_util={avg_str}%")

    with _gpu_budget(budget), _gpu_queue(queue_on_busy):
        # CPU only.
        br = time_batch(items, device="cpu", monitor=True, max_dim=max_dim)
        rows.append(_row("cpu_only", "-", br))
        runs.append(("cpu_only", br))
        print(f"  cpu_only                wall={br.wall_s:.3f}s per_item={br.wall_s/M:.4f}s "
              f"avg_cpu={br.monitor.avg_cpu_pct():.0f}%{_gpu_summary(br)}{_summary(br)}", flush=True)

        # Hybrid with various GPU concurrency caps.
        for cap in args.gpu_caps:
            with _gpu_cap(cap):
                br = time_batch(items, device="gpu", monitor=True, max_dim=max_dim)
            label = f"gpu_cap={cap}" if cap > 0 else "gpu_uncapped"
            rows.append(_row("hybrid", cap, br))
            runs.append((label, br))
            print(f"  {label:<24s} wall={br.wall_s:.3f}s per_item={br.wall_s/M:.4f}s "
                  f"avg_cpu={br.monitor.avg_cpu_pct():.0f}%{_gpu_summary(br)}{_summary(br)}",
                  flush=True)

    path = write_csv("cooperate", list(rows[0].keys()), rows)
    plot_path = plot_cooperate(rows)
    timeline_path = plot_cooperate_timeline(runs)
    print(f"# wrote {path}", flush=True)
    print(f"# wrote {plot_path}", flush=True)
    print(f"# wrote {timeline_path}", flush=True)


def sweep_distribution(args) -> None:
    rng = np.random.default_rng(args.seed)
    n = args.n
    max_dim = args.max_dim
    rows = []
    print(f"# distribution sweep: n={n}, max_dim={max_dim}, distributions={list(DISTRIBUTIONS)}", flush=True)
    for name, gen in DISTRIBUTIONS.items():
        X = gen(n, rng)
        for device in ("cpu", "gpu") if GPU_AVAILABLE else ("cpu",):
            t = time_per_item(X, device=device, max_dim=max_dim)
            rows.append(dict(
                sweep="distribution",
                distribution=name,
                n=n,
                d=X.shape[1],
                max_dim=max_dim,
                device=device,
                median_s=f"{t.median_s:.6f}",
                min_s=f"{t.min_s:.6f}",
            ))
            print(f"  {name:<18s} d={X.shape[1]} max_dim={max_dim} {device:>3s} median={t.median_s:.4f}s", flush=True)
    path = write_csv("distribution", list(rows[0].keys()), rows)
    plot_path = plot_distribution(rows)
    print(f"# wrote {path}", flush=True)
    print(f"# wrote {plot_path}", flush=True)


def sweep_dimensionality(args) -> None:
    rng = np.random.default_rng(args.seed)
    n = args.n
    dims = args.dims
    max_dim = args.max_dim
    rows = []
    print(f"# dimensionality sweep: n={n}, max_dim={max_dim}, dims={dims}", flush=True)
    for d in dims:
        X = gen_gaussian(n, d, rng)
        for device in ("cpu", "gpu") if GPU_AVAILABLE else ("cpu",):
            t = time_per_item(X, device=device, max_dim=max_dim)
            rows.append(dict(
                sweep="dimensionality",
                distribution="gaussian",
                n=n,
                d=d,
                max_dim=max_dim,
                device=device,
                median_s=f"{t.median_s:.6f}",
                min_s=f"{t.min_s:.6f}",
            ))
            print(f"  d={d:>3d} max_dim={max_dim} {device:>3s} median={t.median_s:.4f}s", flush=True)
    path = write_csv("dimensionality", list(rows[0].keys()), rows)
    plot_path = plot_dimensionality(rows)
    print(f"# wrote {path}", flush=True)
    print(f"# wrote {plot_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_n_arg(p, default: int):
    p.add_argument("--n", type=int, default=default, help="point cloud size")


def _add_seed(p):
    p.add_argument("--seed", type=int, default=0, help="numpy rng seed")


def _add_max_dim(p):
    p.add_argument("--max-dim", type=int, default=1,
                   help="persistent homology max dimension (default 1)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("crossover", help="Q1: per-item CPU vs GPU as a function of n")
    p.add_argument("--n-values", type=int, nargs="+", default=None)
    _add_max_dim(p)
    _add_seed(p)
    p.set_defaults(func=sweep_crossover)

    p = sub.add_parser("cpucap", help="Q2: per-batch wall as CPU worker count varies")
    p.add_argument("--n-values", type=int, nargs="+", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    _add_max_dim(p)
    _add_seed(p)
    p.set_defaults(func=sweep_cpucap)

    p = sub.add_parser("cooperate", help="Q3: batch wall with fixed n, varying GPU concurrency cap")
    _add_n_arg(p, 2500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gpu-caps", type=int, nargs="+", default=[1, 2, 4, 8, 0])
    p.add_argument("--budget", type=float, default=None,
                   help="GPU memory budget fraction (0 < f <= 1, default scheduler default 0.6)")
    p.add_argument("--queue-on-busy", action="store_true",
                   help="queue waiting GPU items instead of CPU fallback")
    _add_max_dim(p)
    _add_seed(p)
    p.set_defaults(func=sweep_cooperate)

    p = sub.add_parser("distribution", help="Q4: per-item CPU vs GPU per distribution")
    _add_n_arg(p, 1000)
    _add_max_dim(p)
    _add_seed(p)
    p.set_defaults(func=sweep_distribution)

    p = sub.add_parser("dimensionality", help="Q5: per-item CPU vs GPU per ambient dim")
    _add_n_arg(p, 1000)
    p.add_argument("--dims", type=int, nargs="+", default=[2, 3, 5, 10, 20])
    _add_max_dim(p)
    _add_seed(p)
    p.set_defaults(func=sweep_dimensionality)

    p = sub.add_parser("all", help="run every sweep with default parameters")
    _add_max_dim(p)
    _add_seed(p)
    p.set_defaults(func=lambda a: (sweep_crossover(a), sweep_distribution(a),
                                   sweep_dimensionality(a), sweep_cooperate(a),
                                   sweep_cpucap(a)))

    p = sub.add_parser("render", help="rebuild index.html from existing CSV/PNG outputs")
    p.set_defaults(func=lambda a: print(f"# wrote {render_index_html()}", flush=True))

    args = parser.parse_args(argv)

    # Defaults for sweeps invoked via `all`.
    if args.cmd == "all":
        for k, v in dict(n_values=None, n=1000, batch_size=16,
                         gpu_caps=[1, 2, 4, 8, 0], dims=[2, 3, 5, 10, 20],
                         budget=None, queue_on_busy=False).items():
            if not hasattr(args, k):
                setattr(args, k, v)

    print(f"# build_type={mpsys.build_type()} GPU_AVAILABLE={GPU_AVAILABLE}", flush=True)
    args.func(args)
    if args.cmd != "render":
        print(f"# index: {RESULTS_DIR / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
