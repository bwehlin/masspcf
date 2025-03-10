=====================
Installing masspcf
=====================

Installing using pip
=====================

The easiest way to get started with `masspcf` is using `pip`. There are two packages available: ``masspcf`` and ``masspcf-cpu``. The ``masspcf-cpu`` package is a CPU-only version of the library and is provided as a prebuilt binary wheel.

To get ``masspcf-cpu``, simply run ``pip install masspcf-cpu`` in your Python environment.

The ``masspcf`` package must be built from source. For this to work, one needs to have a C++17 compiler and a CUDA toolkit installed (see https://developer.nvidia.com/cuda-toolkit). After the prerequisites have been satisfied, run ``pip install masspcf``.

.. note:: On OSX, both `masspcf` and `masspcf-cpu` are CPU-only.

.. tip:: If you have a CUDA-capable GPU but are unsure whether the full ``masspcf`` package will build on your system, just try it! If it doesn't work, you can always revert to use ``masspcf-cpu`` instead.

Git source build
=====================

In addition to the released versions of `masspcf` that are available on `PyPI`, it is also possible to install `masspcf` using `Git`. ::

    git clone https://github.com/bwehlin/masspcf.git
    cd masspcf

    # optional: select a specific tagged version to build
    git checkout tags/v0.3.1

    pip install .


On Windows and Linux, `masspcf` is built with CUDA enabled by default. To override this behavior, please set the environment variable ``BUILD_WITH_CUDA=0``. To force a CUDA build, set ``BUILD_WITH_CUDA=1``.