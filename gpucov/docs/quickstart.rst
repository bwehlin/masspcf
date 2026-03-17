Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install gpucov

This pulls in ``libclang`` automatically. No CUDA toolkit is required on the
machine that runs the instrumenter --- GPUCov ships its own minimal CUDA stubs
for parsing.

The CUDA toolkit **is** required on the machine that *builds* and *runs* the
instrumented code.


Step 1: Instrument
------------------

Point GPUCov at your CUDA source files:

.. code-block:: bash

   gpucov instrument \
       --source-root . \
       --output-dir build/_gpucov \
       --files src/kernels.cu include/kernels.cuh \
       -I include

This creates a **shadow directory** (``build/_gpucov/``) containing:

- Instrumented copies of each file, with ``gpucov::hit(N)`` calls injected
  before every executable line in ``__global__`` and ``__device__`` functions.
- ``gpucov_runtime.cuh`` --- the device-side counter array and ``atexit``
  dump handler.
- ``mapping.json`` --- maps each counter ID to a source file and line number.


Step 2: Build
-------------

Compile the instrumented sources instead of the originals. The simplest way
is to prepend the shadow directory to your include path and replace the ``.cu``
source:

.. code-block:: bash

   nvcc -I build/_gpucov -I build/_gpucov/include \
       -DGPUCOV_ENABLED=1 -DGPUCOV_MAX_COUNTERS=64 \
       build/_gpucov/src/kernels.cu \
       -o my_test

Or use the :doc:`CMake integration <cmake>` which handles all of this
automatically.


Step 3: Run tests
-----------------

Set the ``GPUCOV_OUTPUT`` environment variable and run your test suite. When
the process exits, the ``atexit`` handler dumps the counter array to the
specified file:

.. code-block:: bash

   GPUCOV_OUTPUT=coverage/cuda_cov.bin ./my_test

You should see:

.. code-block:: text

   gpucov: dumped 64 counters to coverage/cuda_cov.bin


Step 4: Collect results
-----------------------

Convert the binary dump into human-readable reports:

.. code-block:: bash

   gpucov collect \
       --dump coverage/cuda_cov.bin \
       --mapping build/_gpucov/mapping.json \
       --lcov coverage/cuda.info \
       --summary coverage/summary.json

The ``--lcov`` output is standard `lcov format
<https://ltp.sourceforge.net/coverage/lcov/geninfo.1.php>`_ and can be turned
into an HTML report with ``genhtml``:

.. code-block:: bash

   genhtml coverage/cuda.info --output-directory coverage/html


Dual-compilation files
----------------------

Some ``.cuh`` headers are compiled by **both** the host compiler and NVCC
(e.g. files that use ``#ifdef __CUDACC__`` guards). For these files, GPUCov
wraps injected code in ``#ifdef GPUCOV_ENABLED`` so the host compiler
doesn't see device-only symbols.

Pass ``--dual-compilation`` with glob patterns to identify such files:

.. code-block:: bash

   gpucov instrument \
       --source-root . \
       --output-dir build/_gpucov \
       --files include/cuda/kernels.cuh include/shared/ops.cuh \
       --dual-compilation "ops.cuh" "shared_*.cuh"

GPUCov also auto-detects files containing ``#ifdef __CUDACC__`` or
``#ifdef BUILD_WITH_CUDA`` guards.
