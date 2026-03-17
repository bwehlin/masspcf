CLI Reference
=============

GPUCov provides a command-line interface accessible as ``gpucov`` or
``python -m gpucov``.


Global options
--------------

.. code-block:: text

   gpucov [-h] [--cmake-dir] {instrument,collect} ...

``--cmake-dir``
   Print the path to the directory containing ``GPUCovConfig.cmake`` and exit.
   Used by CMake to locate the GPUCov module:

   .. code-block:: bash

      $ gpucov --cmake-dir
      /path/to/site-packages/gpucov/cmake


``gpucov instrument``
---------------------

Parse CUDA source files and produce instrumented copies with coverage counters.

.. code-block:: text

   gpucov instrument --source-root DIR --output-dir DIR --files FILE [FILE ...]
                     [-I PATH [PATH ...]]
                     [--dual-compilation PATTERN [PATTERN ...]]
                     [--extra-args ARG [ARG ...]]

``--source-root DIR``
   *Required.* Project root directory. Instrumented files are placed in
   ``--output-dir`` preserving their path relative to this root.

``--output-dir DIR``
   *Required.* Shadow directory where instrumented sources, ``mapping.json``,
   and ``gpucov_runtime.cuh`` are written.

``--files FILE [FILE ...]``
   *Required.* CUDA source files to instrument (``.cu`` and ``.cuh``).

``-I``, ``--include-paths PATH [PATH ...]``
   Additional include paths for the libclang parser. Pass the same ``-I``
   paths your build system uses.

``--dual-compilation PATTERN [PATTERN ...]``
   Glob patterns for files compiled by both the host compiler and NVCC.
   Instrumented code in matching files is wrapped in ``#ifdef GPUCOV_ENABLED``
   guards.

``--extra-args ARG [ARG ...]``
   Extra arguments passed directly to the libclang parser.

**Example:**

.. code-block:: bash

   gpucov instrument \
       --source-root . \
       --output-dir build/_gpucov \
       -I include -I 3rd/cub \
       --dual-compilation "shared_ops.cuh" \
       --files \
           src/cuda/kernels.cu \
           include/cuda/kernels.cuh \
           include/shared/shared_ops.cuh

**Output files** (in ``--output-dir``):

``mapping.json``
   Counter-to-source mapping. Structure:

   .. code-block:: json

      {
        "num_counters": 46,
        "mappings": [
          {"id": 0, "file": "/abs/path/to/kernels.cuh", "line": 42},
          {"id": 1, "file": "/abs/path/to/kernels.cuh", "line": 43}
        ]
      }

``gpucov_runtime.cuh``
   Device-side counter infrastructure, automatically ``#include``-d by
   instrumented files.

``<relative/path/to/file>``
   Instrumented copies, mirroring the source tree relative to
   ``--source-root``.


``gpucov collect``
------------------

Read a binary counter dump and produce coverage reports.

.. code-block:: text

   gpucov collect --dump FILE --mapping FILE
                  [--lcov FILE] [--summary FILE]

``--dump FILE``
   *Required.* Path to the binary counter dump written by the runtime
   ``atexit`` handler (controlled by the ``GPUCOV_OUTPUT`` environment
   variable).

``--mapping FILE``
   *Required.* Path to ``mapping.json`` from the instrument step.

``--lcov FILE``
   Write an `lcov .info
   <https://ltp.sourceforge.net/coverage/lcov/geninfo.1.php>`_ file. This is
   the industry-standard format understood by ``genhtml``, Codecov, Coveralls,
   and most coverage aggregation tools.

``--summary FILE``
   Write a JSON summary with aggregate and per-file line coverage:

   .. code-block:: json

      {
        "line_percent": 89.1,
        "lines_total": 46,
        "lines_covered": 41,
        "files": {
          "/path/to/kernels.cuh": {
            "lines_total": 40,
            "lines_covered": 39,
            "line_percent": 97.5
          }
        }
      }

At least one of ``--lcov`` or ``--summary`` must be provided.

**Example:**

.. code-block:: bash

   gpucov collect \
       --dump coverage/cuda_cov.bin \
       --mapping build/_gpucov/mapping.json \
       --lcov coverage/cuda.info \
       --summary coverage/summary.json


Environment variables
---------------------

``GPUCOV_OUTPUT``
   Set at **runtime** (when running the instrumented binary). The ``atexit``
   handler dumps the counter array to this path. If unset or empty, no dump
   is written.

``ENABLE_CUDA_COVERAGE``
   Set at **configure time** (when running CMake). When set to ``1``, the
   GPUCov CMake module sets ``GPUCOV_ENABLE=ON``.
