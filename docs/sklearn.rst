=========================
Scikit-learn integration
=========================

The ``masspcf.sklearn`` module provides transformer classes that wrap
masspcf operations for use in scikit-learn pipelines. Each transformer
handles one step of the topological feature extraction workflow and is
compatible with :class:`~sklearn.pipeline.Pipeline`,
:class:`~sklearn.model_selection.GridSearchCV`, and other sklearn
utilities.

Scikit-learn is not installed automatically with masspcf. Install it
separately::

   pip install scikit-learn

Transformers
============

Five transformers cover the typical TDA pipeline:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Transformer
     - Wraps
     - Input / Output
   * - :py:class:`~masspcf.sklearn.TimeDelayEmbedding`
     - :py:func:`~masspcf.embed_time_delay`
     - NumPy array or ``TimeSeriesTensor`` |rarr| ``PointCloudTensor``
   * - :py:class:`~masspcf.sklearn.PersistentHomology`
     - :py:func:`~masspcf.compute_persistent_homology`
     - ``PointCloudTensor`` |rarr| ``BarcodeTensor``
   * - :py:class:`~masspcf.sklearn.StableRank`
     - :py:func:`~masspcf.barcode_to_stable_rank`
     - ``BarcodeTensor`` |rarr| ``PcfTensor``
   * - :py:class:`~masspcf.sklearn.Mean`
     - :py:func:`~masspcf.mean`
     - ``PcfTensor`` |rarr| ``PcfTensor`` (reduced)
   * - :py:class:`~masspcf.sklearn.PcfKernelTransformer`
     - :py:func:`~masspcf.l2_kernel`
     - ``PcfTensor`` |rarr| NumPy kernel matrix

.. |rarr| unicode:: U+2192


Building a pipeline
===================

A complete classification pipeline chains these transformers with a
scikit-learn estimator::

   from sklearn.pipeline import Pipeline
   from sklearn.svm import SVC
   from masspcf.sklearn import (
       TimeDelayEmbedding,
       PersistentHomology,
       StableRank,
       Mean,
       PcfKernelTransformer,
   )

   pipe = Pipeline([
       ("embed", TimeDelayEmbedding(
           dimension=2, delay=0.3, time_step=0.1,
           window=3.0, stride=1.5)),
       ("ph", PersistentHomology(max_dim=1)),
       ("sr", StableRank(dim=1)),
       ("mean", Mean()),
       ("kernel", PcfKernelTransformer()),
       ("svc", SVC(kernel="precomputed")),
   ])

   pipe.fit(X_train, y_train)
   pipe.score(X_test, y_test)

Each step passes masspcf tensor types to the next. Only the
:class:`~masspcf.sklearn.PcfKernelTransformer` produces a standard
NumPy array (the kernel matrix), which is what the final estimator
expects.


TimeDelayEmbedding
------------------

:class:`~masspcf.sklearn.TimeDelayEmbedding` converts time series
into point clouds via time delay embedding. It accepts either:

- A NumPy array of shape ``(n_instances, n_channels, n_times)``
  (channels-first), which is automatically converted to a
  ``TimeSeriesTensor`` using the ``time_step`` parameter.
- A ``TimeSeriesTensor`` directly.

::

   embed = TimeDelayEmbedding(dimension=3, delay=0.5, time_step=0.1)
   clouds = embed.fit_transform(X)  # PointCloudTensor

Use the ``window`` and ``stride`` parameters to split each recording
into overlapping time windows, producing multiple point clouds per
instance.


PersistentHomology
------------------

:class:`~masspcf.sklearn.PersistentHomology` computes persistent
homology on each point cloud in the tensor::

   ph = PersistentHomology(max_dim=1)
   barcodes = ph.fit_transform(clouds)  # BarcodeTensor

The ``max_dim`` parameter controls the highest homology dimension
computed. The output tensor has an additional last axis of size
``max_dim + 1`` (one barcode per homology dimension).


StableRank
----------

:class:`~masspcf.sklearn.StableRank` converts barcodes to stable rank
PCFs, with optional dimension selection::

   # Extract H1 stable ranks
   sr = StableRank(dim=1)
   features = sr.fit_transform(barcodes)  # PcfTensor

Parameters:

- ``dim``: select a specific homology dimension from the last axis
  (e.g. ``1`` for H1).


Mean
----

:class:`~masspcf.sklearn.Mean` computes the pointwise mean of a PCF
tensor along a specified axis. This is useful for collapsing a window
axis after computing stable ranks::

   mean = Mean()           # reduces along the last axis
   reduced = mean.fit_transform(sranks)  # PcfTensor

Parameters:

- ``dim``: tensor axis to reduce. Defaults to the last axis.


PcfKernelTransformer
--------------------

:class:`~masspcf.sklearn.PcfKernelTransformer` computes the
:math:`L_2` kernel matrix. During ``fit`` it stores the training
features; during ``transform`` it computes the cross-kernel between
new data and the training data::

   kernel = PcfKernelTransformer()
   K_train = kernel.fit_transform(train_features)  # (n, n) symmetric
   K_test = kernel.transform(test_features)          # (m, n) cross-kernel

Use with any estimator that accepts ``kernel="precomputed"``.


Hyperparameter search
=====================

All transformer parameters are accessible through the standard
sklearn ``get_params()`` / ``set_params()`` interface. This makes
``GridSearchCV`` work out of the box::

   from sklearn.model_selection import GridSearchCV

   param_grid = {
       "embed__dimension": [2, 3],
       "embed__delay": [0.2, 0.3, 0.5],
       "embed__window": [2.0, 3.0, 5.0],
   }

   grid = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy")
   grid.fit(X_train, y_train)

   print(grid.best_params_)
   print(grid.score(X_test, y_test))

For a complete worked example, see the
:doc:`motion classification tutorial <tutorial_notebooks/motion_classification>`.
