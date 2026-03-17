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
       ├── gpucov_runtime.cuh   (counter infrastructure + GPUCOV_HIT macro)
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

For each executable line, a ``GPUCOV_HIT(N);`` macro call is inserted
immediately before the line, preserving indentation. An
``#include "gpucov_runtime.cuh"`` directive is added after the existing
include block.

The instrumented source looks like:

.. code-block:: cpp

   #include "gpucov_runtime.cuh"

   __device__ void my_kernel(float* data, int n)
   {
       GPUCOV_HIT(0);
       int idx = blockDim.x * blockIdx.x + threadIdx.x;
       GPUCOV_HIT(1);
       if (idx < n)
       {
           GPUCOV_HIT(2);
           data[idx] = data[idx] + 1;
       }
   }


Phase 2: Runtime counters
--------------------------

``gpucov_runtime.cuh`` provides the counter array, the ``GPUCOV_HIT`` macro,
and an automatic dump handler.

**The macro:**

.. code-block:: cpp

   // Under NVCC:
   #define GPUCOV_HIT(id) gpucov::hit(id)

   // Under any other compiler:
   #define GPUCOV_HIT(id) ((void)0)

This means files compiled by both the host compiler and NVCC (e.g. headers
with ``__host__ __device__`` functions) work automatically --- no special
flags or configuration needed.

**The counter array and hit function:**

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

- **Static device array** --- at 46 counters that's 184 bytes, trivial for
  any GPU. ``GPUCOV_MAX_COUNTERS`` is set to the exact count by the CMake
  module (no wasteful default).
- **One** ``atomicAdd`` **per instrumented line** --- the overhead is
  acceptable for debug/coverage builds (not meant for production).
- **Host+device** ``hit()`` --- compiles for both. On the host side
  (no ``__CUDA_ARCH__``), it's a no-op.
- **atexit auto-dump** --- a static ``AutoDump`` object registers an
  ``atexit`` handler that copies the device counter array to host memory and
  writes it to the path in ``$GPUCOV_OUTPUT``. No explicit API call needed.
- **Per-process dumps** --- ``GPUCOV_OUTPUT`` supports ``%p`` which is
  replaced with the process PID, so multiple test processes can dump
  independently and results are merged at collection time.


Phase 3: Collection
-------------------

The collector reads one or more binary dumps (header: ``uint32 num_counters``,
then ``num_counters * uint32`` values), sums them element-wise, and combines
the result with ``mapping.json`` to produce per-line hit counts.

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
- **Single counter array per process** --- if multiple GPUs are used, all
  threads write to the same device-side array via ``atomicAdd``, which is
  correct. Multiple processes should use ``%p`` in ``GPUCOV_OUTPUT`` and
  merge at collection time.
