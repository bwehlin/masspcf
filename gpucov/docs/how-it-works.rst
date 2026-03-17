How It Works
============

GPUCov operates in three phases: **instrument**, **execute**, and **collect**.


.. code-block:: text

   .cu/.cuh source files
           |
           v   [gpucov instrument + libclang]
   build/_gpucov/
       ├── include/.../*.cuh    (instrumented copies)
       ├── src/.../*.cu         (instrumented copies)
       ├── gpucov_runtime.cuh   (counter infrastructure)
       └── mapping.json         (counter ID -> file:line)
           |
           v   [NVCC via CMake]
   libmycuda.so / my_test      (binary with counters)
           |
           v   [test execution -> atexit handler]
   cuda_cov.bin                 (raw counter array)
           |
           v   [gpucov collect]
   cuda.info (lcov) + summary.json + HTML report


Phase 1: Instrumentation
-------------------------

The instrumenter uses **libclang** to parse CUDA source files as an AST.

**Parsing setup:**

- The file is parsed as ``-x cuda`` with ``--cuda-gpu-arch=sm_70``,
  ``-nocudalib``, ``-nocudainc``.
- GPUCov ships minimal ``cuda_runtime.h`` stubs so parsing works without
  the CUDA toolkit installed.
- System C++ standard library paths are auto-detected from ``g++`` so
  ``<vector>``, ``<functional>``, etc. resolve.

**Device function detection:**

The AST is walked to find function declarations (``FUNCTION_DECL``,
``FUNCTION_TEMPLATE``, ``CXX_METHOD``) in the target file. For each, the raw
source text around the declaration is searched for ``__global__`` or
``__device__`` attributes.

**Executable line collection:**

Within each device function's body (``COMPOUND_STMT``), the AST is walked for
executable statement nodes:

- ``IF_STMT``, ``FOR_STMT``, ``WHILE_STMT``, ``DO_STMT``
- ``RETURN_STMT``, ``CALL_EXPR``
- ``BINARY_OPERATOR``, ``COMPOUND_ASSIGNMENT_OPERATOR``, ``UNARY_OPERATOR``
- ``DECL_STMT``, ``SWITCH_STMT``, ``CASE_STMT``
- ``MEMBER_REF_EXPR``

Lines where insertion would break syntax (``else``, ``}``, ``case``,
``default:``) are filtered out.

**Source rewriting:**

For each executable line, a ``gpucov::hit(N);`` call is inserted immediately
before the line, preserving indentation. The ``#include "gpucov_runtime.cuh"``
directive is added after the existing include block.

For dual-compilation files (compiled by both host and NVCC), injected code is
wrapped in ``#ifdef GPUCOV_ENABLED`` guards.


Phase 2: Runtime counters
--------------------------

``gpucov_runtime.cuh`` defines:

.. code-block:: cpp

   namespace gpucov {
       __device__ unsigned int g_counters[GPUCOV_MAX_COUNTERS];

       __host__ __device__ __forceinline__ void hit(unsigned int id) {
   #ifdef __CUDA_ARCH__
           if (id < GPUCOV_MAX_COUNTERS)
               atomicAdd(&g_counters[id], 1u);
   #endif
       }
   }

Key design choices:

- **Static device array** --- 8 KB at the default 2048 counters. Trivial
  for any GPU.
- **One** ``atomicAdd`` **per instrumented line** --- the overhead is
  acceptable for debug/coverage builds (not meant for production).
- **Host+device** ``hit()`` --- the function compiles for both host and
  device. On the host side (no ``__CUDA_ARCH__``), it's a no-op. This allows
  dual-compilation files to call ``hit()`` without errors.
- **atexit auto-dump** --- a static ``AutoDump`` object registers an
  ``atexit`` handler that copies the device counter array to host memory and
  writes it to the path in ``$GPUCOV_OUTPUT``. No explicit API call needed.


Phase 3: Collection
-------------------

The collector reads the binary dump (header: ``uint32 num_counters``, then
``num_counters * uint32`` values) and combines it with ``mapping.json`` to
produce per-line hit counts.

**Output formats:**

lcov ``.info``
   Industry-standard format. Each source file gets a record:

   .. code-block:: text

      TN:CUDA Coverage
      SF:/path/to/kernels.cuh
      DA:42,15
      DA:43,0
      LF:40
      LH:39
      end_of_record

   Compatible with ``genhtml``, Codecov, Coveralls, and other tools.

JSON summary
   Aggregate and per-file line coverage percentages. Designed for
   dashboards and CI badge generation.


Shadow directory
----------------

Instrumented files live in a shadow directory (e.g. ``build/_gpucov/``).
Original sources are never modified.

The CMake module (or manual build setup) prepends the shadow directory to the
include search path so the compiler finds the instrumented ``.cuh`` headers
before the originals. Non-instrumented sibling files are symlinked into the
shadow tree so that relative ``#include`` paths (e.g. ``../task.h``) still
resolve.

The ``.cu`` translation unit is replaced entirely --- the CMake module swaps
the original source in the target's ``SOURCES`` property with the instrumented
copy.


Limitations
-----------

- **CUDA only** --- currently parses with ``-x cuda``. Other GPU languages
  (HIP, SYCL) are not yet supported but the architecture is extensible.
- **Template definitions, not instantiations** --- counters are placed in
  template function bodies. All instantiations share the same counter IDs.
- **No branch coverage** --- only line-level hit counts. A line with an ``if``
  shows whether it was reached, not which branch was taken.
- **Single counter dump** --- if multiple processes or multiple GPUs write
  to the same ``GPUCOV_OUTPUT`` path, the last writer wins. Run tests
  sequentially or use per-process output paths and merge the results.
