========
Sampling
========

The :py:mod:`masspcf.sampling` module provides distance-weighted sampling from
point clouds. Given a source point cloud and a set of vantage points, the
sampler draws *k* points around each vantage point, with probabilities
controlled by a weight function of distance. This is useful for constructing local
neighborhoods, subsampling large point clouds, and building stochastic
approximations in TDA pipelines.

All sampling operations support deterministic seeding via
:py:class:`~masspcf.random.Generator` (see :doc:`random`).


Mathematical setup
==================

Let :math:`\mathcal{X} = \{x_1, \dots, x_N\} \subset \mathbb{R}^D` be a
source point cloud and :math:`v \in \mathbb{R}^D` a vantage point. A
*weight function* :math:`g` assigns a non-negative weight to each
point based on its distance to the vantage point:

.. math::

   w_i = g\!\left(\|x_i - v\|\right).

The sampler draws points with probability proportional to :math:`w_i`:

.. math::

   P(x_i) = \frac{g(\|x_i - v\|)}{\sum_{j=1}^{N} g(\|x_j - v\|)}

Only the relative values of :math:`g` matter — scaling :math:`g` by any
positive constant leaves the sampling distribution unchanged. In particular,
there is no requirement that :math:`g` be normalized: an unnormalized
Gaussian kernel :math:`e^{-x^2/2\sigma^2}` and the corresponding PDF
:math:`\frac{1}{\sigma\sqrt{2\pi}} e^{-x^2/2\sigma^2}` produce identical
samples.

This is repeated independently for each vantage point in a set
:math:`\{v_1, \dots, v_M\}`.


Weight functions
================

Sampling weights are defined by *weight functions* — callables that map a
non-negative scalar to a non-negative weight. In the context of
:py:func:`~masspcf.sampling.sample`, the weight function is evaluated at
the Euclidean distance from each source point to the vantage point, but the
functions themselves are generic and make no assumption about what the input
represents.

Three built-in weight functions are provided.

Gaussian
--------

.. math::

   g(x) = \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)

Peaked around :math:`x = \mu` with width :math:`\sigma`. When used for
distance-based sampling, setting :math:`\mu = 0` gives the familiar
RBF-style weighting where nearby points are strongly preferred.

::

   from masspcf.sampling import Gaussian

   # Favor points near the vantage (distance ~ 0)
   dist = Gaussian(mean=0.0, sigma=1.0)

   # Favor points at distance ~ 3 from the vantage
   dist = Gaussian(mean=3.0, sigma=0.5)

Uniform
-------

.. math::

   g(x) = \begin{cases}
     1 & \text{if } l \le x \le h, \\
     0 & \text{otherwise.}
   \end{cases}

Returns 1 for inputs in the interval :math:`[l, h]` and 0 outside. When
used for distance-based sampling, this selects points in an annular
neighborhood.

::

   from masspcf.sampling import Uniform

   # Sample from an annulus at distance 1–2
   dist = Uniform(lo=1.0, hi=2.0)

Mixture
--------

.. math::

   g(x) = \sum_{j=1}^{C} w_j \, g_j(x)

where each :math:`g_j` is any weight function. Components can be
Gaussian, Uniform, or even nested Mixtures. Supports multi-modal weighting
— for example, sampling from both a local neighborhood and a more distant
shell.

::

   from masspcf.sampling import Gaussian, Uniform, Mixture

   # Mix of two Gaussians
   dist = Mixture(
       components=[Gaussian(mean=0.0, sigma=0.5),
                   Gaussian(mean=5.0, sigma=0.5)],
       weights=[0.7, 0.3],
   )

   # Mix of a Gaussian and a Uniform ring
   dist = Mixture(
       components=[Gaussian(mean=0.0, sigma=1.0),
                   Uniform(lo=3.0, hi=5.0)],
       weights=[0.8, 0.2],
   )


Basic usage
===========

The main entry point is :py:func:`~masspcf.sampling.sample`. It takes a source
point cloud, a set of vantage points, a sample count *k*, and a weight
function.

::

   import numpy as np
   import masspcf as mpcf
   from masspcf.sampling import Gaussian, sample

   # Source: 1000 points in R^3
   pts = np.random.randn(1000, 3)
   X = mpcf.zeros((1,), dtype=mpcf.pcloud64)
   X[0] = pts

   # Vantage points: 50 query locations
   queries = np.random.randn(50, 3)
   V = mpcf.zeros((1,), dtype=mpcf.pcloud64)
   V[0] = queries

   # 30 samples per vantage, Gaussian weighting
   dist = Gaussian(mean=0.0, sigma=1.0)
   result = sample(X, V, k=30, dist=dist)

   # result is a PointCloudTensor of shape (50,)
   # result[i] is a (30, 3) point cloud sampled around queries[i]

The output is a :py:class:`~masspcf.tensor.PointCloudTensor` where each element
contains the *k* sampled points for that vantage.


Sampling with and without replacement
======================================

By default, sampling is **with replacement** — the same source point may
appear multiple times in a single vantage's sample::

   result = sample(X, V, k=30, dist=dist, replace=True)   # default

To sample **without replacement**, set ``replace=False``::

   result = sample(X, V, k=30, dist=dist, replace=False)

Without replacement, each source point appears at most once per vantage.
If *k* exceeds the number of points with non-zero weight, the sampler
will return as many unique points as possible (up to *k*).


Ball restriction
================

An optional ``radius`` parameter restricts sampling to points within a ball
of the given radius around each vantage point::

   result = sample(X, V, k=30, dist=dist, radius=2.0)

Points further than ``radius`` from the vantage are never sampled, regardless
of the distribution. This is useful when the distribution has broad or heavy
tails but you want to enforce a hard distance cutoff. Internally, the
restriction is applied by pruning KD-tree subtrees whose minimum distance
to the vantage exceeds the radius — no separate code path is needed.


Reproducibility
===============

Pass a :py:class:`~masspcf.random.Generator` for deterministic results::

   from masspcf.random import Generator

   gen = Generator(seed=42)
   result = sample(X, V, k=30, dist=dist, generator=gen)

The same seed produces the same samples regardless of thread count or
execution order. See :doc:`random` for details.


Bias–efficiency trade-off
==========================

The sampler is unbiased by default: accepted samples are drawn with
probability exactly proportional to :math:`g(d)`. However, for distributions
with well-separated modes and small bandwidths (e.g., a Mixture of tight
Gaussians centered at distances 0 and 5), the unbiasing correction can make
the acceptance rate very low for distant modes.

The ``min_correction`` parameter controls this trade-off:

- ``0.0`` (default) — exact unbiased sampling. The effective distribution
  is exactly proportional to :math:`g(d)`. May have low acceptance rates
  for well-separated modes.
- ``1.0`` — no correction. Points on heavily-tightened tree paths are
  overweighted, but all modes are efficiently reachable.
- Values in between — points with correction factor above the threshold
  are sampled exactly; only points on heavily-tightened paths are
  overweighted. Clamping never underweights any point.

::

   # Tight mixture that would have near-zero acceptance for the far mode
   dist = Mixture(
       components=[Gaussian(mean=0.0, sigma=0.5),
                   Gaussian(mean=5.0, sigma=0.5)],
       weights=[0.5, 0.5],
   )

   # Allow bias to ensure both modes are reachable
   result = sample(X, V, k=100, dist=dist, min_correction=1.0)

See :doc:`internals/tree_importance_sampling` for a full probabilistic
analysis of the clamping effect.


Precision
=========

Both ``pcloud32`` and ``pcloud64`` are supported. The dtype is inferred
from the source tensor, or can be specified explicitly::

   result = sample(X, V, k=30, dist=dist, dtype=mpcf.pcloud32)
