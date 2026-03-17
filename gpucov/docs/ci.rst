CI Integration
==============

GPUCov is designed to run in CI pipelines on GPU-equipped runners.


GitHub Actions
--------------

A typical workflow adds coverage to an existing CUDA test job:

.. code-block:: yaml

   cuda_test:
     runs-on: [self-hosted, gpu, cuda12]
     env:
       ENABLE_CUDA_COVERAGE: 1
       GPUCOV_OUTPUT: ${{ github.workspace }}/covreport/cuda/cuda_cov.bin
     steps:
       - uses: actions/checkout@v4
         with:
           submodules: recursive

       - name: Install gpucov
         run: pip install gpucov

       - name: Configure
         run: cmake -B build -DCMAKE_BUILD_TYPE=Debug

       - name: Build
         run: cmake --build build -j $(nproc)

       - name: Run tests
         run: |
           mkdir -p covreport/cuda
           cd test && ../build/my_test

       - name: Collect CUDA coverage
         run: |
           gpucov collect \
             --dump covreport/cuda/cuda_cov.bin \
             --mapping build/_gpucov_my_cuda/mapping.json \
             --lcov covreport/cuda/cuda.info \
             --summary covreport/cuda/summary.json

       - name: Generate HTML report
         run: genhtml covreport/cuda/cuda.info -o covreport/cuda/html

       - name: Upload artifact
         uses: actions/upload-artifact@v4
         with:
           name: cuda-coverage
           path: covreport/cuda/

Key points:

- ``ENABLE_CUDA_COVERAGE=1`` activates the CMake module at configure time.
- ``GPUCOV_OUTPUT`` tells the runtime where to write the counter dump.
- The mapping file path depends on the target name:
  ``build/_gpucov_<target>/mapping.json``.
- ``genhtml`` (from the ``lcov`` package) produces the HTML report. Install
  it with ``apt-get install lcov``.


Merging with host coverage
--------------------------

The lcov output can be merged with host-side coverage (from gcov or llvm-cov)
using ``lcov --add-tracefile``:

.. code-block:: bash

   # Merge host + GPU coverage into a single report
   lcov --add-tracefile covreport/cpp/cpp.info \
        --add-tracefile covreport/cuda/cuda.info \
        --output-file covreport/combined.info

   genhtml covreport/combined.info -o covreport/combined_html


Docker / self-hosted runners
----------------------------

If your CI runner uses a Docker image, ensure ``gpucov`` is pre-installed:

.. code-block:: dockerfile

   FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
   RUN pip install gpucov

Or add it to your runner's ``requirements.txt``. GPUCov's only Python
dependency is ``libclang``, which is a pure wheel (~25 MB).


Codecov / Coveralls
-------------------

The lcov ``.info`` file is directly compatible with Codecov and Coveralls
upload tools:

.. code-block:: yaml

   - name: Upload to Codecov
     uses: codecov/codecov-action@v4
     with:
       files: covreport/cuda/cuda.info
       flags: cuda
