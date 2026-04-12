Tutorials
=====================

.. toctree::
   :hidden:

   tutorial_notebooks/masspcf_intro_mnist_vis
   tutorial_notebooks/pytorch_tda_classifier
   tutorial_notebooks/lorenz_takens_embedding
   tutorial_notebooks/timeseries_classification

:doc:`Visualizing the space of handwritten digits using masspcf <tutorial_notebooks/masspcf_intro_mnist_vis>`
   Getting started with masspcf and doing basic topological data visualization on the MNIST dataset.
   |dl_mnist|

This notebook is described in the following video tutorial from the `Applied Algebraic Topology Research Network <https://aatrn.net>`_'s Spring 2025 Tutorial-a-thon:

.. youtube :: 3pTJH3T9G74

(Note: masspcf is not affiliated with Applied Algebraic Topology Research Network.)

:doc:`Topological features for PyTorch classifiers <tutorial_notebooks/pytorch_tda_classifier>`
   Using masspcf to compute topological feature vectors from point clouds and training a PyTorch neural network to classify them.
   |dl_pytorch|

:doc:`Takens embedding of the Lorenz attractor <tutorial_notebooks/lorenz_takens_embedding>`
   Simulating the Lorenz system, storing it as a multi-channel time series, and using time delay embedding to reconstruct the attractor from a single channel.
   |dl_lorenz|

:doc:`Time series classification with persistent homology <tutorial_notebooks/timeseries_classification>`
   Classifying dynamical regimes by counting statistically significant topological features in windowed time-delay embeddings.
   |dl_tsclass|

.. |dl_mnist| replace:: :download:`Download notebook <tutorial_notebooks/masspcf_intro_mnist_vis.ipynb>`
.. |dl_pytorch| replace:: :download:`Download notebook <tutorial_notebooks/pytorch_tda_classifier.ipynb>`
.. |dl_lorenz| replace:: :download:`Download notebook <tutorial_notebooks/lorenz_takens_embedding.ipynb>`
.. |dl_tsclass| replace:: :download:`Download notebook <tutorial_notebooks/timeseries_classification.ipynb>`