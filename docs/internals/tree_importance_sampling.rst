==============================
Tree-based importance sampling
==============================

masspcf's distance-weighted sampler uses the KD-tree built over the source
point cloud as a hierarchical proposal distribution. This page describes the
algorithm in detail, explains why it adapts to any distance distribution
without strategy switching, and analyses its correctness and complexity.


Background and prior art
=========================

The algorithm belongs to the family of **hierarchical importance sampling**
methods that use spatial tree structures as proposal distributions. The core
idea — traversing a tree top-down by choosing children proportional to
contribution upper bounds, then correcting with accept/reject at the leaf —
originates in the computer graphics literature.

**Lightcuts** [WABG06]_ introduced this approach for efficiently sampling
from large collections of light sources during rendering. For each shading
point, a tree over lights is traversed by bounding each node's contribution
from the geometric relationship (distance, orientation) to the shading point.
**Stochastic Lightcuts** [Yuk19]_ refined the method with improved bound
estimation and stratification. The same principle has appeared in stochastic
variants of **Barnes–Hut** N-body methods and in **tree-based fast
summation** for kernel density estimation [GM03]_.

masspcf adapts this approach to distance-weighted point cloud sampling: the
"lights" are source points, the "shading point" is the vantage point, and the
"contribution" is the distance distribution weight. The accept/reject
correction at the leaf is standard von Neumann rejection [vN51]_.


Problem statement
==================

Given:

- A source point cloud :math:`\mathcal{X} = \{x_1, \dots, x_N\} \subset \mathbb{R}^D`.
- A vantage point :math:`v \in \mathbb{R}^D`.
- A distance distribution :math:`g : [0, \infty) \to [0, \infty)` defining the weight of each point: :math:`w_i = g(\|x_i - v\|)`.
- A sample count :math:`k`.

We want to draw :math:`k` points from :math:`\mathcal{X}` with probability proportional to :math:`w_i`, without evaluating :math:`g` at all :math:`N` points.

The naive approach (compute all weights, build a prefix sum, binary search) costs :math:`O(N)` per vantage point. With :math:`M` vantage points this gives :math:`O(NM)` total — prohibitive when both :math:`N` and :math:`M` are large, especially when the weight distribution is peaked and most of the :math:`O(N)` work is wasted on near-zero weights.


Why a tree?
============

A KD-tree partitions :math:`\mathbb{R}^D` into axis-aligned regions. Each internal node covers a spatial region with a known bounding box, and each leaf contains a small number of points (up to a configurable leaf size, defaulting to 10).

For a given vantage point :math:`v`, we can compute the **distance range** :math:`[d_{\min}, d_{\max}]` from :math:`v` to any node's bounding box. This gives an upper bound on the weight of any point in that node:

.. math::

   W_{\text{upper}}(\text{node}) = n_{\text{node}} \cdot \max_{d \in [d_{\min}, d_{\max}]} g(d)

where :math:`n_{\text{node}}` is the number of points in the subtree. This bound is computable in :math:`O(D)` time (one distance calculation per dimension), without touching any individual point.

The key insight is that these per-node bounds form a **hierarchical proposal distribution**: at every level of the tree, we can choose which child to descend into with probability proportional to their weight bounds, progressively focusing on the regions where the weight mass actually is.


Algorithm
==========

The sampler proceeds in two phases: tree construction (once per source cloud) and per-sample traversal (repeated for each of the :math:`k \cdot M` samples needed).


Phase 1: Tree construction and precomputation
----------------------------------------------

1. Build a KD-tree over the source point cloud using nanoflann [BL14]_. The tree uses the :math:`L_2` metric and a leaf size of 10. Construction is :math:`O(N \log N)`.

2. Precompute subtree sizes for every node via a single post-order traversal. For each leaf, the size is the number of points it contains (stored directly in the nanoflann node). For each internal node, the size is the sum of its children's sizes. These counts are stored in a hash map keyed by node pointer. This traversal is :math:`O(N)`.

3. Extract the root bounding box from nanoflann (stored as ``root_bbox_``), giving the axis-aligned bounding box of all points.


Phase 2: Per-sample tree traversal
------------------------------------

To draw one sample for vantage point :math:`v`:

1. **Start at the root.** Initialize the current bounding box to the root bounding box.

2. **At each internal node**, compute bounding boxes for the left and right children by splitting the current box along the node's split dimension:

   - Left child: upper bound on split dimension is set to ``divlow`` (the maximum value in the left subtree along the split dimension).
   - Right child: lower bound on split dimension is set to ``divhigh`` (the minimum value in the right subtree along the split dimension).

3. **Compute distance ranges** from :math:`v` to each child's bounding box:

   .. math::

      d_{\min}^2 = \sum_{i=1}^D \max(0,\; l_i - v_i,\; v_i - h_i)^2

   .. math::

      d_{\max}^2 = \sum_{i=1}^D \max(|v_i - l_i|,\; |v_i - h_i|)^2

   where :math:`[l_i, h_i]` is the bounding box interval in dimension :math:`i`. These are the squared minimum and maximum distances from :math:`v` to any point in the box.

4. **Compute weight upper bounds** for each child:

   .. math::

      W_L = n_L \cdot g_{\max}(d_{\min}^L, d_{\max}^L), \qquad
      W_R = n_R \cdot g_{\max}(d_{\min}^R, d_{\max}^R)

   where :math:`g_{\max}(a, b) = \max_{d \in [a, b]} g(d)` is provided by the distribution's ``max_in_range`` method, and :math:`n_L, n_R` are the precomputed subtree sizes.

   If a radius restriction is active, any child whose :math:`d_{\min}^2` exceeds :math:`r^2` gets :math:`W = 0` and is never visited.

5. **Descend** into the left child with probability :math:`W_L / (W_L + W_R)`, or the right child otherwise. Record the chosen child's weight :math:`W_c` as ``w_self`` for the next level.

6. **Update the correction factor.** At each non-root internal node, the sum :math:`S = W_L + W_R` of the children's refined bounds is generally less than ``w_self`` (the bound the parent assigned to this node), because the children have tighter bounding boxes. Accumulate:

   .. math::

      \text{correction} \;\leftarrow\; \text{correction} \times \frac{S}{\text{w\_self}}

   At the root (first node visited), ``w_self`` is not yet defined, so this step is skipped.

7. **At a leaf node**, pick a point uniformly at random from the :math:`n_{\text{leaf}}` points in the leaf. Apply one final correction for the leaf-level tightening:

   .. math::

      \text{correction} \;\leftarrow\; \text{correction} \times \frac{n_{\text{leaf}} \cdot g_{\max}(d_{\min}^{\text{leaf}}, d_{\max}^{\text{leaf}})}{\text{w\_self}}

8. **Accept/reject.** Compute the actual distance :math:`d = \|x - v\|` and the actual weight :math:`g(d)`. Accept the point with probability:

   .. math::

      p_{\text{accept}} = \frac{g(d)}{g_{\max}(d_{\min}^{\text{leaf}}, d_{\max}^{\text{leaf}})} \times \text{correction}

   If rejected, restart from step 1 with a fresh correction factor.

9. For **without-replacement** sampling, maintain a set of accepted point indices and reject duplicates.


Correctness
============

The traversal implicitly defines a proposal distribution :math:`q(x)` over source points. Without the correction factor, the acceptance probability would be path-dependent: at each tree level, the children's refined bounds are tighter than the parent's estimate, and the "tightening ratio" :math:`S/W` varies across the tree. This makes the denominator of the acceptance probability depend on *which* path through the tree was taken to reach a point, not just on the point itself.

The correction factor eliminates this path dependence via a **telescoping argument**. Consider the probability of reaching a specific leaf :math:`\ell` and selecting point :math:`x` within it. The traversal probability is:

.. math::

   q(\ell) = \prod_{k} \frac{W_{c_k}}{S_k}

where the product is over all internal nodes on the root-to-leaf path, :math:`c_k` is the chosen child at level :math:`k`, and :math:`S_k = W_L^{(k)} + W_R^{(k)}`. Within the leaf, the point is selected uniformly with probability :math:`1/n_\ell`.

The cumulative correction factor along this path is:

.. math::

   \text{correction} = \prod_{k} \frac{S_{k+1}}{W_{c_k}}

where :math:`S_{k+1}` is the children's sum at the next level (or the leaf bound for the last factor). This product telescopes with :math:`q(\ell)`:

.. math::

   q(\ell) \times \text{correction} = \frac{1}{S_{\text{root}}} \times n_\ell \cdot g_{\max}(\text{leaf})

so the overall single-trial acceptance probability for point :math:`x` becomes:

.. math::

   p_{\text{accept}}(x) = q(\ell) \times \frac{1}{n_\ell} \times \frac{g(d_x)}{g_{\max}(\text{leaf})} \times \text{correction} = \frac{g(d_x)}{S_{\text{root}}}

The denominator :math:`S_{\text{root}}` is the **same constant** for all points. Therefore, accepted samples are drawn with probability exactly proportional to :math:`g(d_x)` — the sampling is unbiased [vN51]_.

**Why the correction stays in** :math:`[0, 1]`: At every node, children's refined bounds are tighter, so :math:`S \leq W` and each factor :math:`S/W \leq 1`. The cumulative product is therefore :math:`\leq 1`. Combined with :math:`g(d)/g_{\max} \leq 1`, the acceptance probability is always valid.


Bias–efficiency trade-off: ``min_correction``
-----------------------------------------------

For distributions with well-separated modes and small bandwidths, the
correction factor can compound to near-zero along paths to distant modes.
This makes the sampler unbiased but impractical: nearly all proposals for the
far mode are rejected.

The ``min_correction`` parameter (denoted :math:`c` below) clamps the
cumulative correction to :math:`[c, 1]` at each step during traversal.


Effect on the sampling distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each point :math:`x` in the source cloud has a unique root-to-leaf path
through the KD-tree. Let :math:`\gamma(x)` denote the unclamped correction
factor for that path — the product of all tightening ratios :math:`S/W`
encountered during traversal. Without clamping, the telescoping argument
gives an acceptance probability of :math:`g(d_x) / S_{\text{root}}` for
every point. The effective sampling distribution is:

.. math::

   P_0(x) \;\propto\; g(d_x)

With clamping at :math:`c`, the correction for point :math:`x` becomes
:math:`\max(\gamma(x),\, c)` instead of :math:`\gamma(x)`. The effective
sampling distribution is:

.. math::

   P_c(x) \;\propto\; g(d_x) \cdot \frac{\max(\gamma(x),\, c)}{\gamma(x)}
         \;=\; g(d_x) \cdot \max\!\left(1,\; \frac{c}{\gamma(x)}\right)

This has two regimes:

- **Unclamped points** (:math:`\gamma(x) \geq c`): :math:`P_c(x) \propto g(d_x)`. No distortion — these points are sampled with the exact target weights.

- **Clamped points** (:math:`\gamma(x) < c`): :math:`P_c(x) \propto g(d_x) \cdot c / \gamma(x)`. These points are **overweighted** relative to the exact distribution, by a factor of :math:`c / \gamma(x) > 1`.

Clamping never underweights any point. The distortion is one-sided: clamped
points (those on heavily-tightened tree paths) are overrepresented, while
unclamped points retain their exact relative weights.


When does clamping activate?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The correction :math:`\gamma(x)` is small when the tree traversal passes
through nodes where the children's refined weight bounds are much tighter
than the parent's estimate — i.e., where the bounding box split dramatically
reduces the distance range. This typically happens when:

- The distribution has well-separated modes (e.g., a Mixture of tight
  Gaussians at distances 0 and 5).
- The tree splits the near and far clusters into separate subtrees.
- The path to the far cluster accumulates many tightening factors < 1.

For unimodal or broad distributions, :math:`\gamma(x)` rarely drops below
0.5 and clamping has no practical effect.


Choosing ``min_correction``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :math:`c = 0` (default): exact, unbiased sampling. May be impractical
  for well-separated modes.
- :math:`c = 1`: no correction at all; the effective distribution is
  :math:`P_1(x) \propto g(d_x) / \gamma(x)`, which overweights points in
  parts of the tree with loose bounds. This is the most efficient setting
  and appropriate when approximate proportionality suffices.
- Intermediate values: trade off between bias and efficiency. All points
  with :math:`\gamma(x) \geq c` are sampled exactly; only points on
  heavily-tightened paths are affected.


``max_in_range`` implementations
=================================

The tightness of the weight bounds — and hence the acceptance rate — depends on how well ``max_in_range`` approximates the true maximum of :math:`g` over the interval :math:`[d_{\min}, d_{\max}]`.

**Gaussian** :math:`g(d) = \exp(-(d - \mu)^2 / 2\sigma^2)`:

- If :math:`\mu \in [d_{\min}, d_{\max}]`: returns 1 (the peak is inside the interval).
- Otherwise: returns :math:`\max(g(d_{\min}), g(d_{\max}))` — the maximum is at the closer endpoint since the Gaussian is unimodal.

This is exact for the Gaussian.

**Uniform** :math:`g(d) = \mathbf{1}[l \leq d \leq h]`:

- If the intervals :math:`[d_{\min}, d_{\max}]` and :math:`[l, h]` overlap: returns 1.
- Otherwise: returns 0.

Also exact.

**Mixture** :math:`g(d) = \sum_j w_j g_j(d)`:

- Returns :math:`\sum_j w_j \cdot g_j^{\max}(d_{\min}, d_{\max})`.

This is a conservative upper bound (the component maxima may not occur at the same :math:`d`), but it is fast to compute and reasonably tight when the components are well-separated.


Complexity
===========

**Tree construction:** :math:`O(N \log N)` for building the KD-tree, plus :math:`O(N)` for the subtree size precomputation. This is done once per source cloud.

**Per-sample cost:** Each traversal visits :math:`O(\log N)` nodes (one per level of the tree). At each node, the work is :math:`O(D)` for bounding box distance computations. So each sample attempt costs :math:`O(D \log N)`.

**Acceptance rate:** Depends on how tight the weight bounds are. For smooth, unimodal distributions (e.g. Gaussian), the bounds are tight and the acceptance rate is high — typically above 50%. For distributions with sharp features or the conservative mixture bound, more rejections occur. The expected number of attempts per accepted sample is :math:`1 / p_{\text{accept}}`, where :math:`p_{\text{accept}}` is the average acceptance probability.

**Total cost per vantage point:** :math:`O(k \cdot D \log N / p_{\text{accept}})`.

**Total cost for all vantage points:** :math:`O(M \cdot k \cdot D \log N / p_{\text{accept}})`, parallelized across vantage points via Taskflow.

Compare with the naive approach: :math:`O(N \cdot M)` to compute all weights. The tree-based approach wins when :math:`k \cdot D \log N / p_{\text{accept}} \ll N`, which is typical when the distribution is peaked (most weight mass is in a small region) or when :math:`k \ll N`.


Adaptive behavior
===================

The algorithm automatically adapts to the shape of the distance distribution:

**Peaked distributions** (e.g., Gaussian with small :math:`\sigma`): The weight bounds decay rapidly away from the vantage point. Subtrees far from :math:`v` get near-zero bounds and are almost never visited. The traversal concentrates on a small spatial neighborhood, effectively behaving like a ball query — but without requiring the user to specify a radius.

**Broad distributions** (e.g., Gaussian with large :math:`\sigma`): Weight bounds are similar across all subtrees. The traversal explores the tree broadly, behaving like near-uniform sampling with an accept/reject correction for the exact distribution shape.

**Heavy-tailed distributions**: Unlike a ball query (which hard-truncates), the tree traversal assigns non-zero probability to all subtrees. Far-away subtrees are visited with frequency proportional to their weight, so the tails of the distribution are faithfully represented.

**Multi-modal distributions** (e.g., a Mixture of Gaussians): Subtrees near each mode receive high weight bounds. The traversal splits its probability mass across the modes proportionally, sampling from all of them without requiring the user to enumerate them.


Ball restriction interaction
=============================

When a ``radius`` is specified, nodes whose minimum distance to :math:`v` exceeds the radius receive a weight bound of zero. This prunes entire subtrees from the traversal without any additional code path. The effect is equivalent to setting :math:`g(d) = 0` for :math:`d > r`, but more efficient because the pruning happens at the node level rather than the point level.


Parallelism
============

The tree is built once and shared (read-only) across all vantage points. Each vantage point's sampling is independent, so the outer loop over vantage points is parallelized via ``parallel_walk`` and Taskflow. Each vantage point receives its own deterministically-seeded random engine (see :doc:`deterministic_random`), ensuring reproducible results regardless of thread count.


Implementation files
=====================

- ``include/mpcf/sampling/distribution.hpp`` — distribution types and ``DistanceDist`` concept
- ``include/mpcf/sampling/kdtree.hpp`` — nanoflann adaptor for ``PointCloud<T>``
- ``include/mpcf/sampling/sample.hpp`` — tree traversal and sampling algorithm
- ``src/python/sampling/py_sampling.cpp`` — pybind11 bindings
- ``masspcf/sampling.py`` — Python wrapper


References
==========

.. [BL14] Blanco, J. L., & Rai, P. K. (2014). nanoflann: a C++ header-only fork of FLANN, a library for nearest neighbor (NN) with KD-trees. https://github.com/jlblancoc/nanoflann


.. [GM03] Gray, A. G., & Moore, A. W. (2003). Rapid evaluation of multiple density models. *Proceedings of the Ninth International Workshop on Artificial Intelligence and Statistics (AISTATS)*.

.. [vN51] von Neumann, J. (1951). Various techniques used in connection with random digits. *National Bureau of Standards Applied Mathematics Series*, 12, 36–38.

.. [WABG06] Walter, B., Fernandez, S., Arbree, A., Bala, K., Donikian, M., & Greenberg, D. P. (2005). Lightcuts: a scalable approach to illumination. *ACM Transactions on Graphics (SIGGRAPH)*, 24(3), 1098–1107.

.. [Yuk19] Yuksel, C. (2019). Stochastic Lightcuts. *Proceedings of the Conference on High-Performance Graphics*, 27–32.
