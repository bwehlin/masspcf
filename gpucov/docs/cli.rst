CLI Reference
=============

GPUCov provides a command-line interface accessible as ``gpucov`` or
``python -m gpucov``.


Global options
--------------

.. code-block:: text

   gpucov [-h] [--cmake-dir] {instrument,zerocounters,collect} ...

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

``--extra-args ARG [ARG ...]``
   Extra arguments passed directly to the libclang parser.

**Example:**

.. code-block:: bash

   gpucov instrument \
       --source-root . \
       --output-dir build/_gpucov \
       -I include -I 3rd/cub \
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
   Device-side counter infrastructure and the ``GPUCOV_HIT`` macro,
   automatically ``#include``-d by instrumented files.

``<relative/path/to/file>``
   Instrumented copies, mirroring the source tree relative to
   ``--source-root``.


``gpucov zerocounters``
-----------------------

Remove dump files from a previous run, ensuring a clean slate before
re-running tests. Analogous to ``lcov --zerocounters``.

.. code-block:: text

   gpucov zerocounters --dump PATTERN [PATTERN ...]

``--dump PATTERN [PATTERN ...]``
   *Required.* Glob pattern(s) matching dump files to delete. Use the same
   pattern you pass to ``gpucov collect --dump``.

**Example:**

.. code-block:: bash

   # Clear all per-process dumps before re-running tests
   gpucov zerocounters --dump "coverage/cuda_*.bin"

Returns exit code 0 regardless of whether any files were found (idempotent).


``gpucov collect``
------------------

Read binary counter dump(s) and produce coverage reports.

.. code-block:: text

   gpucov collect --dump FILE [FILE ...] --mapping FILE
                  [--lcov FILE] [--summary FILE]

``--dump FILE [FILE ...]``
   *Required.* Path(s) to binary counter dump files written by the runtime
   ``atexit`` handler (controlled by the ``GPUCOV_OUTPUT`` environment
   variable). Supports glob patterns. When multiple files are given, counters
   are summed element-wise --- use this with ``%p`` in ``GPUCOV_OUTPUT`` to
   merge results from multiple test processes.

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

   # Single dump file
   gpucov collect \
       --dump coverage/cuda_cov.bin \
       --mapping build/_gpucov/mapping.json \
       --lcov coverage/cuda.info \
       --summary coverage/summary.json

   # Merge multiple per-process dumps
   gpucov collect \
       --dump "coverage/cuda_*.bin" \
       --mapping build/_gpucov/mapping.json \
       --lcov coverage/cuda.info


Environment variables
---------------------

``GPUCOV_OUTPUT``
   Set at **runtime** (when running the instrumented binary). The ``atexit``
   handler dumps the counter array to this path. If unset or empty, no dump
   is written.

   Supports ``%p`` which is replaced with the process PID, allowing
   multiple test processes to write separate dump files:

   .. code-block:: bash

      GPUCOV_OUTPUT=coverage/cuda_%p.bin ./my_test

``ENABLE_CUDA_COVERAGE``
   Set at **configure time** (when running CMake). When set to ``1``, the
   GPUCov CMake module sets ``GPUCOV_ENABLE=ON``.
