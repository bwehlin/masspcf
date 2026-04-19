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

from . import _mpcf_cpp as cpp


def force_cpu(on: bool):
    """Set forced execution on CPU. By default, execution may happen on either CPU or GPU (if using a GPU-enabled build of masspcf).

    Parameters
    ----------
    on : bool
      If `True`, force execution on CPU for all operations. If `False`, execution may happen on either CPU or GPU (if using a GPU-enabled build of masspcf).
    """
    cpp.force_cpu(on)


def set_block_size(x: int, y: int):
    """Set CUDA block size for (GPU) matrix computations. This is an advanced option that should only be modified by expert users.

    Parameters
    ----------
    x : int
      Horizontal block size

    y : int
      Vertical block size
    """
    cpp.set_block_dim(x, y)


def limit_cpus(n: int):
    """Sets the upper limit on the number of CPU threads that can be used for computations.

    Typically, the default corresponding to the number of hardware CPU threads is a good choice but it can be warranted to limit the number of threads in, e.g., multi-user environments. For normal use, we recommend using the default.

    Parameters
    ----------
    n : int
      Number of CPU threads to use
    """
    cpp.limit_cpus(n)


def limit_gpus(n: int):
    """Sets the number of GPUs that can be used by masspcf. By default, all available GPUs are used.

    This option only has an effect if masspcf is compiled with GPU support.

    Parameters
    ----------
    n : int
      Number of GPUs to use
    """
    cpp.limit_gpus(n)


def set_hybrid_gpu_queue_on_busy(on: bool):
    """Set the hybrid Ripser++ dispatcher's policy when a GPU slot is unavailable.

    By default (``on=False``), items that cannot obtain a GPU reservation --
    because all slots are busy or the memory budget is exhausted -- fall
    back to the CPU Ripser path immediately. With ``on=True`` they instead
    block and wait for a GPU slot to free up, unless the item is
    structurally too large to ever fit on any visible GPU (in which case
    CPU fallback still happens).

    Queue-on-busy is preferable when GPU is much faster than CPU per item
    (``max_dim >= 2`` or large ``n``, where Ripser++'s apparent-pairs
    acceleration dominates). CPU fallback is preferable at low
    ``max_dim``, where CPU is competitive per item and parallel CPU
    throughput outweighs queueing behind fewer GPU slots.

    OOM-triggered CPU fallback (after a real cudaMalloc failure during a
    reserved GPU job) is unconditional regardless of this setting.

    Parameters
    ----------
    on : bool
      If True, wait for a GPU slot instead of falling back to CPU.
    """
    cpp.set_hybrid_gpu_queue_on_busy(on)


def set_gpu_budget_fraction(f: float):
    """Set the fraction of free GPU memory the hybrid persistence dispatcher reserves as its scheduling budget.

    At scheduler construction the dispatcher reads the free memory on each
    visible GPU and multiplies by ``f`` to get its budget. The remainder
    stays unclaimed, absorbing CUDA scratch allocations, fragmentation,
    and other tenants. Raising ``f`` lets the scheduler admit more
    concurrent GPU jobs at the cost of safety headroom; lowering it makes
    the dispatcher more conservative.

    Default is 0.6, which is a reasonable trade-off on single-tenant GPUs.
    Valid range is ``0 < f <= 1``.

    Parameters
    ----------
    f : float
      New budget fraction.
    """
    cpp.set_gpu_budget_fraction(f)


def limit_gpu_concurrency(n: int):
    """Set a cap on the number of concurrent GPU jobs the hybrid persistence dispatcher will run.

    By default (``n = 0``), the dispatcher is bounded only by per-GPU memory: the
    :class:`GpuMemoryScheduler` admits as many concurrent Ripser++ instances as
    fit in the available memory budget. Setting ``n > 0`` adds a hard cap on
    the total number of in-flight GPU jobs across all visible GPUs; further
    items are routed to the CPU until a slot frees up.

    This option only has an effect if masspcf is compiled with GPU support and
    the workload uses the hybrid persistence path (``device="auto"`` or
    ``device="gpu"``).

    Parameters
    ----------
    n : int
      Maximum concurrent GPU jobs across all GPUs. Use ``0`` for no cap (default).
    """
    cpp.limit_gpu_concurrency(n)


def set_cuda_threshold(n: int):
    """Sets how many PCFs are required in a matrix computation before computations are moved from CPU to GPU. By default, the threshold is set to 500 PCFs.

    Parameters
    ----------
    n : int
      Number of PCFs required before (supported) matrix computations are moved to GPU
    """
    cpp.set_cuda_threshold(n)


def set_device_verbose(on: bool):
    """Enable verbose device output. In this mode, when operations that may occur on GPU are invoked, a message is logged stating whether the operation will be performed on CPU or GPU.

    Parameters
    ----------
    on : bool
      Enable verbose device logging
    """
    cpp.set_device_verbose(on)


def set_min_block_side(n: int):
    """Set the minimum block side length for the CUDA block scheduler.

    This controls the minimum number of threads per GPU kernel launch, ensuring
    good GPU occupancy. A value of 0 (the default) auto-detects from the GPU
    hardware (SM count), targeting ~50% max occupancy.

    This is an advanced option that should only be modified by expert users.

    Parameters
    ----------
    n : int
      Minimum block side length. 0 = auto-detect from GPU hardware.
    """
    cpp.set_min_block_side(n)


def build_type() -> str:
    """Return the build type of the masspcf backend.

    Returns
    -------
    str
        ``"CUDA"`` if built with GPU support, ``"CPU"`` otherwise.
    """
    return cpp._build_type()
