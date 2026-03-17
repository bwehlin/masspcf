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

- Instrumented copies of each file, with ``GPUCOV_HIT(N)`` macro calls
  injected before every executable line in ``__global__`` and ``__device__``
  functions.
- ``gpucov_runtime.cuh`` --- the device-side counter array, the
  ``GPUCOV_HIT`` macro, and an ``atexit`` dump handler.
- ``mapping.json`` --- maps each counter ID to a source file and line number.

Files compiled by both the host compiler and NVCC work automatically ---
``GPUCOV_HIT`` expands to a counter increment under NVCC and to nothing
under a host compiler, so no special configuration is needed.


Step 2: Build
-------------

Compile the instrumented sources instead of the originals. The simplest way
is to prepend the shadow directory to your include path and replace the ``.cu``
source:

.. code-block:: bash

   nvcc -I build/_gpucov -I build/_gpucov/include \
       -DGPUCOV_MAX_COUNTERS=64 \
       build/_gpucov/src/kernels.cu \
       -o my_test

``GPUCOV_MAX_COUNTERS`` must match the counter count from the instrument step
(printed to stdout and written to ``mapping.json``).

Or use the :doc:`CMake integration <cmake>` which handles all of this
automatically.


Step 3: Run tests
-----------------

First, clear any dump files left over from a previous run:

.. code-block:: bash

   gpucov zerocounters --dump "coverage/cuda_*.bin"

Then set the ``GPUCOV_OUTPUT`` environment variable and run your test suite.
Use ``%p`` in the path so each process writes its own dump file:

.. code-block:: bash

   GPUCOV_OUTPUT=coverage/cuda_%p.bin ./my_test
   GPUCOV_OUTPUT=coverage/cuda_%p.bin python -m pytest .

At process exit, the ``atexit`` handler dumps the counter array automatically:

.. code-block:: text

   gpucov: dumped 64 counters to coverage/cuda_12345.bin


Step 4: Collect results
-----------------------

Convert the binary dump(s) into human-readable reports:

.. code-block:: bash

   gpucov collect \
       --dump "coverage/cuda_*.bin" \
       --mapping build/_gpucov/mapping.json \
       --lcov coverage/cuda.info \
       --summary coverage/summary.json

``--dump`` accepts multiple files and glob patterns. When multiple dumps are
provided, counters are summed.

The ``--lcov`` output is standard `lcov format
<https://ltp.sourceforge.net/coverage/lcov/geninfo.1.php>`_ and can be turned
into an HTML report with ``genhtml``:

.. code-block:: bash

   genhtml coverage/cuda.info --output-directory coverage/html
