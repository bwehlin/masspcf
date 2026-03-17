GPUCov
======

**Line-level coverage for GPU device code.**

Standard coverage tools (gcov, llvm-cov) only instrument host-side code compiled
by GCC or Clang --- NVCC-compiled GPU kernels are invisible. GPUCov fills this
gap by source-level instrumentation: it parses ``.cu``/``.cuh`` files using
**libclang**, identifies executable lines inside ``__global__`` and ``__device__``
functions, injects ``atomicAdd`` counter increments, and after test execution
collects hit counts and produces **lcov** reports.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      Install, instrument, build, run tests, and view your first coverage
      report in five minutes.

   .. grid-item-card:: CMake Integration
      :link: cmake
      :link-type: doc

      Drop-in CMake module --- two function calls to add GPU coverage to any
      CUDA target.

   .. grid-item-card:: CLI Reference
      :link: cli
      :link-type: doc

      Full reference for the ``gpucov instrument`` and ``gpucov collect``
      commands.

   .. grid-item-card:: How It Works
      :link: how-it-works
      :link-type: doc

      Architecture overview: parsing, instrumentation, the runtime counter
      array, and report generation.

.. toctree::
   :maxdepth: 2
   :hidden:

   quickstart
   cmake
   cli
   how-it-works
   ci
   api
