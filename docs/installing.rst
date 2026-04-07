==========
Installing
==========

Installing using pip
=====================

The easiest way to get started with `masspcf` is using `pip`::

    pip install masspcf

Git source build
=====================

In addition to the released versions of `masspcf` that are available on `PyPI`, it is also possible to install `masspcf` using `Git`. ::

    git clone https://github.com/bwehlin/masspcf.git
    cd masspcf

    # optional: select a specific tagged version to build
    git checkout tags/v0.4.0

    pip install .


On Windows and Linux, `masspcf` is built with CUDA enabled by default. To override this behavior, please set the environment variable ``BUILD_WITH_CUDA=0``. To force a CUDA build, set ``BUILD_WITH_CUDA=1``.

.. note:: On OSX, ``masspcf`` is CPU-only.
