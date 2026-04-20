"""Single hybrid PH call on a real .mpcf file, scoped with NVTX for
nsys filtering. Skips the cooperate sweep / matplotlib / CSV output
of bench_ph_hybrid.py so the trace is tightly focused.

Usage:
  micromamba run -n py310 nsys profile \\
      --trace=cuda,osrt,nvtx --sample=process-tree \\
      --output=benchmarks/bench_results/<tag> --force-overwrite=true \\
      python benchmarks/profile_real.py samples_3_50_2500.mpcf

  micromamba run -n py310 nsys stats --filter-nvtx=ph_compute \\
      --report cuda_api_sum benchmarks/bench_results/<tag>.nsys-rep
"""
from __future__ import annotations

import argparse
import time

import masspcf as mpcf
import masspcf.persistence as mpers
import masspcf.system as mpsys

try:
    import nvtx
except ImportError:
    nvtx = None


def _annotate(name: str):
    if nvtx is None:
        from contextlib import nullcontext
        return nullcontext()
    return nvtx.annotate(name)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="path to a .mpcf point-cloud tensor file")
    p.add_argument("--max-dim", type=int, default=1)
    p.add_argument("--queue-on-busy", action="store_true", default=True)
    p.add_argument("--no-queue-on-busy", dest="queue_on_busy",
                   action="store_false")
    p.add_argument("--gpu-cap", type=int, default=0,
                   help="GPU concurrency cap (0=unlimited)")
    p.add_argument("--warmup", action="store_true", default=True,
                   help="run a small warmup PH call so the timed call "
                        "doesn't include CUDA context / library init")
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    args = p.parse_args()

    X = mpcf.load(args.path)
    print(f"loaded {args.path}: shape={X.shape}, dtype={X.dtype}, "
          f"items={X.size}", flush=True)

    if hasattr(mpsys, "set_hybrid_gpu_queue_on_busy"):
        mpsys.set_hybrid_gpu_queue_on_busy(args.queue_on_busy)
    if hasattr(mpsys, "limit_gpu_concurrency"):
        mpsys.limit_gpu_concurrency(args.gpu_cap)

    if args.warmup:
        _ = mpers.compute_persistent_homology(
            X[:1], max_dim=args.max_dim, device="gpu")

    t0 = time.perf_counter()
    with _annotate("ph_compute"):
        mpers.compute_persistent_homology(
            X, max_dim=args.max_dim, device="gpu")
    wall = time.perf_counter() - t0

    print(f"wall={wall:.3f}s items={X.size} per_item={wall/X.size:.4f}s "
          f"max_dim={args.max_dim} queue_on_busy={args.queue_on_busy}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
