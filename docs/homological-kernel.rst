===========================
The 0-th Homological Kernel
===========================

This guide covers how to use the 0-th homological kernel in stablebear. The
method takes in point clouds or distance matrices and returns persistence
barcodes, which can then be used in the normal persistent homology pipeline
(see :doc:`persistence`).

Background
==========

The 0-th homological kernel was first developed with a specific use case in
mind: using TDA to build a new correlation function that could stand in for a
classical measure such as Pearson correlation. That original idea took a point
cloud in :math:`\mathbb{R}^2` and projected its points onto the diagonal line.
The persistence barcodes of both the original cloud and its diagonal projection
were computed, and the *kernel* between them was extracted. The result is a
barcode that resembles an :math:`H_1` barcode -- its bars do not always start at
zero. This score is highly correlated with Pearson correlation, but is better at
detecting non-linear correlation and clustering in the data. When the points are
centered and normalized, a perfectly linear relationship projects onto the
diagonal unchanged, so the :math:`L_1` integral of the resulting stable rank is
zero; the integral then grows with the amount of non-linearity in the data. This
construction was used successfully to analyze protein correlations in ALS
patients.

That original idea later evolved, once the result was generalized. All that the
method needs is a set :math:`X` and two distances :math:`d` and :math:`d'` with
:math:`d \ge d'` for all points in :math:`X`. This gives two distance spaces
:math:`(X, d)` and :math:`(X, d')`, and the kernel of the induced map on 0-th
homology can be computed for any such pair -- making it possible to work with
data of any dimension and with distances other than the diagonal projection.
These higher-dimensional uses do not have as clean an interpretation as the
correlation case. The simplest way to read the output is as 0-th homology
mimicking 1-st homology: it exposes the difference in topology between two
distance spaces built on the same points.

The generalization, and the proof of correctness for the algorithm below, come
from forthcoming work by the authors -- an approved master's thesis that is not
yet published.


What the kernel measures
========================

Because :math:`d' \le d`, the identity on points is 1-Lipschitz as a map
:math:`(X, d) \to (X, d')`, so it induces a map between the Vietoris-Rips
filtrations and, after applying the 0-th homology functor, a map of persistence
modules

.. math::

   \mu \colon H_0(\mathrm{VR}(X, d)) \longrightarrow H_0(\mathrm{VR}(X, d')) .

At every scale :math:`t`, the finer distance :math:`d'` has already connected at
least as many points as :math:`d`, so :math:`\mu_t` is surjective: it sends each
:math:`d`-component to the :math:`d'`-component that contains it. It is generally
not injective -- two points can be joined under :math:`d'` while still separate
under :math:`d`. The **0-th homological kernel** is precisely that failure of
injectivity,

.. math::

   \ker \mu = \ker\bigl( H_0(\mathrm{VR}(X, d)) \to H_0(\mathrm{VR}(X, d')) \bigr) ,

a persistence module that decomposes into a barcode
:math:`\bigoplus_i [w_i, v_i)`. Each bar tracks one component that is identified
early under :math:`d'` but stays separate longer under :math:`d`:

- the **birth** :math:`w_i` is the scale at which two components merge under the
  finer distance :math:`d'` -- the moment the discrepancy appears;
- the **death** :math:`v_i` is the scale at which those same components merge
  under the coarse distance :math:`d` -- the moment the discrepancy is resolved.

Since :math:`d' \le d`, every death is at least its birth, so all bars are
well-formed. A **zero-length bar** :math:`[w_i, w_i)` means the two distances
merge those components at the very same scale: no disagreement there. Long bars
mark where the two structures genuinely diverge. A set of :math:`n` points always
produces :math:`n - 1` bars, one per merge in the hierarchy.


How the kernel is computed
==========================

The kernel could be read off the two full persistence modules, but that is
expensive and indirect. Instead the algorithm exploits a single structural fact:
if the merges are processed **in the order dictated by the finer distance**
:math:`d'`, smallest merge scale first, then at every step exactly one bar of the
kernel splits off as a direct summand. This reduces the whole computation to
single-linkage hierarchical clustering plus a running distance update.

Concretely, ``stablebear`` follows these steps:

1. **Cophenetic distances.** Replace :math:`d` and :math:`d'` by the
   single-linkage cophenetic (sub-dominant ultra-pseudo-metric) distances
   :math:`d_{sd}` and :math:`d'_{sd}`. The cophenetic distance between two points
   is the scale at which they first land in the same connected component, so this
   step encodes the entire 0-th persistence of each filtration.

2. **Merge order from** :math:`d'`. Extract the :math:`n - 1` merges of the
   :math:`d'`-hierarchy in non-decreasing scale order. The :math:`i`-th merge
   joins components :math:`[a_i]` and :math:`[b_i]` at birth
   :math:`w_i = d'_{sd}([a_i], [b_i])`, with
   :math:`w_1 \le w_2 \le \dots \le w_{n-1}`.

3. **Deaths from** :math:`d`. Maintain a distance :math:`d^{(0)} = d_{sd}` on the
   components. For :math:`i = 1, \dots, n - 1`:

   a. read the death :math:`v_i = d^{(i-1)}([a_i], [b_i])` -- the scale at which
      the same two components merge under the current coarse distance;
   b. record the kernel bar :math:`[w_i, v_i)`;
   c. contract :math:`[a_i]` and :math:`[b_i]` into a single component;
   d. update the coarse distance by the single-linkage (quotient) rule: for every
      remaining component :math:`c`,

      .. math::

         d^{(i)}([a_i] \cup [b_i],\, c)
           = \min\bigl(d^{(i-1)}([a_i], c),\, d^{(i-1)}([b_i], c)\bigr).

The result is the kernel barcode
:math:`\ker \mu \cong \bigoplus_{i=1}^{n-1} [w_i, v_i)`.

Two subtleties are worth highlighting, because the test suite guards both:

- **Deaths are a property of the whole component pair, not of the merging edge.**
  The death :math:`v_i` is the coarse-distance merge scale of the two *components*
  :math:`[a_i]` and :math:`[b_i]`, which can be realized by a pair of points that
  touches neither endpoint of the edge that triggered the :math:`d'`-merge. The
  running quotient distance in step 3(d) is what keeps this correct.
- **Ties are order-independent.** When several merges share a birth scale, the
  resulting barcode is the same multiset regardless of which tied merge is
  processed first.

If the input violates :math:`d' \le d` -- some merge is "born" after it would
have to die -- the computation cannot produce a valid bar and raises a
``RuntimeError``.


Computing the kernel in stablebear
==================================

:py:func:`~stablebear.persistence.compute_homological_kernel` takes the coarse
data :math:`d` as ``X`` and the finer data :math:`d'` as ``X_prime``, and returns
a :py:class:`~stablebear.persistence.BarcodeTensor`. The finer distance is
supplied in one of two ways.

Built-in projection presets
---------------------------

For the correlation use case, pass a point cloud and let a ``transform`` preset
produce :math:`d'` by projecting the points. Every preset is an orthogonal
projection, so the domination :math:`d' \le d` is guaranteed for any input::

   import numpy as np
   from stablebear import persistence

   # A small 2D "correlation cloud"
   X = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [3.0, 3.0]])

   bcs = persistence.compute_homological_kernel(X, transform="diagonal")
   kernel = bcs[0]     # one Barcode, with len(X) - 1 = 3 bars

The available presets are:

- ``"diagonal"`` -- projection onto the diagonal line spanned by
  :math:`(1, \dots, 1)`. This is the correlation construction: it compares the
  cloud with its collapse onto :math:`y = x`.
- ``"coordinate"`` -- projection onto the first :math:`\dim - 1` coordinate axes
  (drops the last coordinate).

Explicit ``X_prime``
--------------------

To use a projection or transform of your own, pass the finer data directly as
``X_prime``. It must be the same kind and shape as ``X``, and element :math:`i`
of ``X`` is paired with element :math:`i` of ``X_prime``. Point clouds stay in
the ambient dimension -- express a lower-dimensional projection in the original
coordinates rather than dropping columns::

   X = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [3.0, 3.0]])

   # Diagonal projection expressed explicitly: (x, y) -> ((x+y)/2, (x+y)/2)
   m = X.mean(axis=1, keepdims=True)
   X_prime = np.broadcast_to(m, X.shape).copy()

   bcs = persistence.compute_homological_kernel(X, X_prime)

Exactly one of ``X_prime`` and ``transform`` must be given; supplying both or
neither is a ``ValueError``.

Distance-matrix input
---------------------

When the metrics are not Euclidean point clouds -- for instance abstract
distances where :math:`d'` reorders the merges relative to :math:`d` -- provide
them as distance matrices. ``transform`` presets require point-cloud input, so
distance matrices must always be paired with an explicit ``X_prime``::

   import stablebear as sb
   from stablebear import persistence

   d  = sb.DistanceMatrix(3, dtype=sb.float64)
   dp = sb.DistanceMatrix(3, dtype=sb.float64)
   # ... fill d and dp with d' <= d entrywise ...

   bcs = persistence.compute_homological_kernel(d, dp)

Accepted input types
--------------------

Both ``X`` and ``X_prime`` accept the same forms as
:py:func:`~stablebear.persistence.compute_persistent_homology`:

- a plain NumPy array or ``FloatTensor`` (a single point cloud);
- a ``DistanceMatrix`` (a single precomputed metric);
- a ``PointCloudTensor`` or ``DistanceMatrixTensor`` (a whole tensor of them,
  computed in parallel).

A single-instance input (array, ``FloatTensor``, or ``DistanceMatrix``) returns a
``BarcodeTensor`` of shape ``(1,)``. A tensor input returns a barcode tensor of
the same shape, one kernel barcode per element::

   # A batch of correlation clouds, one kernel each
   clouds = sb.zeros((100,), dtype=sb.pcloud64)
   for i in range(100):
       clouds[i] = np.random.randn(30, 2)

   kernels = persistence.compute_homological_kernel(clouds, transform="diagonal")
   print(kernels.shape)   # (100,)


From kernel to correlation score
================================

The kernel barcode is an ordinary :py:class:`~stablebear.persistence.Barcode`,
so it feeds straight into the functional summaries from :doc:`persistence`. In
the correlation use case, the score is the total persistence of the kernel,
obtained as the :math:`L_1` integral of its stable rank::

   import stablebear as sb
   from stablebear import persistence

   kernels = persistence.compute_homological_kernel(clouds, transform="diagonal")

   # Stable rank of each kernel, then its L1 norm = total persistence
   sranks = persistence.barcode_to_stable_rank(kernels)
   scores = sb.lp_norm(sranks, p=1)

A score of zero means every bar has zero length: the cloud and its projection
merge identically, i.e. the dependency is linear. Larger scores mark stronger
non-linear structure. Because the summaries are PCFs, the resulting tensors
support the usual distances, means, and norms -- so a whole cohort of clouds can
be compared, averaged, or classified with the same tools used elsewhere in
``stablebear``.
