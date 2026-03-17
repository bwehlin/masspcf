"""GPUCov — line-level coverage for GPU device code.

Provides source-level instrumentation of __global__/__device__ functions
via atomicAdd counters, producing lcov-compatible coverage reports.

No existing tool provides line-level coverage for CUDA device code.
Standard coverage tools (gcov, llvm-cov) only instrument host-side code.
GPUCov solves this by parsing .cu/.cuh files with libclang, identifying
executable lines inside device functions, and injecting counter increments.
"""

__version__ = "0.1.0"
