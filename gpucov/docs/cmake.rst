CMake Integration
=================

GPUCov ships a CMake module that reduces integration to a few lines. The module
handles instrumentation, shadow directory setup, source replacement, include
path manipulation, and compile definitions.


Loading the module
------------------

GPUCov's CMake module is installed as package data. Use ``gpucov --cmake-dir``
to locate it:

.. code-block:: cmake

   # After find_package(Python ...) or find_package(Python3 ...)
   execute_process(
       COMMAND "${Python_EXECUTABLE}" -m gpucov --cmake-dir
       OUTPUT_VARIABLE _GPUCOV_CMAKE_DIR
       OUTPUT_STRIP_TRAILING_WHITESPACE
       ERROR_QUIET
       RESULT_VARIABLE _GPUCOV_CMAKE_RESULT
   )
   if (_GPUCOV_CMAKE_RESULT EQUAL 0)
     include("${_GPUCOV_CMAKE_DIR}/GPUCovConfig.cmake")
   endif ()

After inclusion, two variables are available:

- ``GPUCOV_FOUND`` --- ``TRUE`` if ``gpucov`` is importable by the Python
  interpreter.
- ``GPUCOV_ENABLE`` --- cache variable, ``ON``/``OFF``. Controls whether
  ``gpucov_instrument_target`` actually instruments anything.


Enabling coverage
-----------------

Coverage is **off by default**. Enable it with either:

- **Environment variable** (recommended for CI):

  .. code-block:: bash

     ENABLE_CUDA_COVERAGE=1 cmake -B build

- **CMake cache variable**:

  .. code-block:: bash

     cmake -B build -DGPUCOV_ENABLE=ON

The environment variable takes precedence over a cached value, so CI can
toggle coverage without clearing the CMake cache.


Instrumenting a target
----------------------

.. code-block:: cmake

   if (GPUCOV_FOUND)
     gpucov_instrument_target(my_cuda_lib
         FILES
             src/cuda/kernels.cu
             include/cuda/kernels.cuh
             include/shared/ops.cuh
         INCLUDE_PATHS
             "${CMAKE_SOURCE_DIR}/include"
             "${CMAKE_SOURCE_DIR}/3rd/some_lib"
     )
   endif ()

When ``GPUCOV_ENABLE`` is ``ON``, this:

1. Runs the instrumenter on the listed ``FILES``.
2. Creates a shadow directory at ``${CMAKE_BINARY_DIR}/_gpucov_<target>``.
3. Replaces ``.cu`` sources in the target with instrumented copies.
4. Symlinks non-instrumented files into the shadow tree so relative
   ``#include`` paths still resolve.
5. Prepends shadow include directories so instrumented ``.cuh`` files take
   precedence over originals.
6. Defines ``GPUCOV_ENABLED=1`` and ``GPUCOV_MAX_COUNTERS=<N>`` on the
   target.

Files compiled by both the host compiler and NVCC work automatically ---
the ``GPUCOV_HIT`` macro expands to a counter increment under NVCC and
to nothing under a host compiler.

When ``GPUCOV_ENABLE`` is ``OFF``, the function returns immediately and the
target is untouched.


Function signature
------------------

.. code-block:: cmake

   gpucov_instrument_target(<target>
       FILES <file1> [<file2> ...]
       [SOURCE_ROOT <dir>]
       [INCLUDE_PATHS <path1> [<path2> ...]]
       [EXTRA_ARGS <arg1> [<arg2> ...]]
   )

``<target>``
   The CMake target to instrument (must already exist via ``add_library`` or
   similar).

``FILES``
   *Required.* CUDA source files to instrument (``.cu`` and ``.cuh``).
   Relative paths are resolved against ``SOURCE_ROOT``.

``SOURCE_ROOT``
   Project root for computing relative paths. Defaults to
   ``CMAKE_SOURCE_DIR``.

``INCLUDE_PATHS``
   Additional ``-I`` paths passed to the libclang parser.

``EXTRA_ARGS``
   Extra arguments passed directly to the libclang parser (e.g. ``-DFOO=1``).


Exported variables
------------------

After ``gpucov_instrument_target`` succeeds, two variables are set in the
calling scope:

- ``GPUCOV_<target>_SHADOW_DIR`` --- path to the shadow directory.
- ``GPUCOV_<target>_MAPPING`` --- path to ``mapping.json`` (needed by
  ``gpucov collect``).


Complete example
----------------

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.24)
   project(myproject LANGUAGES CXX CUDA)

   find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)

   # Load GPUCov (no-op if not installed)
   execute_process(
       COMMAND "${Python_EXECUTABLE}" -m gpucov --cmake-dir
       OUTPUT_VARIABLE _GPUCOV_CMAKE_DIR
       OUTPUT_STRIP_TRAILING_WHITESPACE
       ERROR_QUIET RESULT_VARIABLE _GPUCOV_CMAKE_RESULT
   )
   if (_GPUCOV_CMAKE_RESULT EQUAL 0)
     include("${_GPUCOV_CMAKE_DIR}/GPUCovConfig.cmake")
   endif ()

   # CUDA library
   add_library(my_cuda STATIC src/kernels.cu)
   target_include_directories(my_cuda PUBLIC include)

   # Instrument (only when GPUCOV_ENABLE is ON)
   if (GPUCOV_FOUND)
     gpucov_instrument_target(my_cuda
         FILES src/kernels.cu include/kernels.cuh
         INCLUDE_PATHS "${CMAKE_SOURCE_DIR}/include"
     )
   endif ()

Then build and run:

.. code-block:: bash

   ENABLE_CUDA_COVERAGE=1 cmake -B build
   cmake --build build
   GPUCOV_OUTPUT=cuda_cov.bin ./build/my_test
   gpucov collect --dump cuda_cov.bin \
       --mapping build/_gpucov_my_cuda/mapping.json \
       --lcov cuda.info --summary summary.json
   genhtml cuda.info -o html_report
