#    Copyright 2024-2026 Bjorn Wehlin
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

r"""Distance-weighted sampling from point clouds.

This module provides distance-weighted samplers that draw points from a
source point cloud with probabilities controlled by a weight function.
Internally, a KD-tree over the source cloud serves as a hierarchical proposal
distribution, giving :math:`O(\log N)` per-sample cost regardless of the
weight function shape.

See :doc:`/sampling` for a user guide and :doc:`/internals/tree_importance_sampling`
for algorithmic details.
"""

from . import _mpcf_cpp as cpp
from .tensor import FloatTensor, PointCloudTensor
from .typing import _validate_dtype, float32, float64, pcloud32, pcloud64


class Gaussian:
    r"""Gaussian weight function.

    .. math::

       g(x) = \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)

    Parameters
    ----------
    mean : float
        Centre :math:`\mu` of the Gaussian.
    sigma : float
        Width :math:`\sigma` of the Gaussian.
    """

    def __init__(self, mean: float, sigma: float, dtype=float64):
        if sigma <= 0:
            raise ValueError(f"sigma ({sigma}) must be positive")
        dtype = _validate_dtype(dtype, [float32, float64])
        cls = {float32: cpp.Gaussian32, float64: cpp.Gaussian64}[dtype]
        self._cpp = cls(mean, sigma)

    @property
    def mean(self):
        return self._cpp.mean

    @property
    def sigma(self):
        return self._cpp.sigma

    def __call__(self, x: float) -> float:
        return self._cpp(x)

    def max(self) -> float:
        return self._cpp.max()

    def max_in_range(self, x_min: float, x_max: float) -> float:
        return self._cpp.max_in_range(x_min, x_max)


class Uniform:
    r"""Uniform weight function.

    Returns 1 for inputs in :math:`[lo, hi]` and 0 otherwise.

    Parameters
    ----------
    lo : float
        Lower bound of the support interval.
    hi : float
        Upper bound of the support interval.
    dtype : type, optional
        ``float32`` or ``float64``, by default ``float64``.
    """

    def __init__(self, lo: float, hi: float, dtype=float64):
        if lo > hi:
            raise ValueError(f"lo ({lo}) must be <= hi ({hi})")
        dtype = _validate_dtype(dtype, [float32, float64])
        cls = {float32: cpp.Uniform32, float64: cpp.Uniform64}[dtype]
        self._cpp = cls(lo, hi)

    @property
    def lo(self):
        return self._cpp.lo

    @property
    def hi(self):
        return self._cpp.hi

    def __call__(self, x: float) -> float:
        return self._cpp(x)

    def max(self) -> float:
        return self._cpp.max()

    def max_in_range(self, x_min: float, x_max: float) -> float:
        return self._cpp.max_in_range(x_min, x_max)


class Mixture:
    r"""Weighted mixture of component weight functions.

    .. math::

       g(x) = \sum_{j=1}^{C} w_j \, g_j(x)

    Each component can be any weight function (Gaussian, Uniform, or
    another Mixture).

    Parameters
    ----------
    components : list of weight functions
        The component functions (e.g. Gaussian, Uniform).
    weights : list of float
        Non-negative mixing weights (need not sum to 1).
    dtype : type, optional
        ``float32`` or ``float64``, by default ``float64``.
    """

    def __init__(self, components: list, weights: list[float], dtype=float64):
        self.components = components
        self.weights = weights
        dtype = _validate_dtype(dtype, [float32, float64])
        mix_cls = {float32: cpp.Mixture32, float64: cpp.Mixture64}[dtype]
        self._cpp = mix_cls([c._cpp for c in components], weights)

    def __call__(self, x: float) -> float:
        return self._cpp(x)

    def max(self) -> float:
        return self._cpp.max()

    def max_in_range(self, x_min: float, x_max: float) -> float:
        return self._cpp.max_in_range(x_min, x_max)


_PCLOUD_TO_FLOAT = {pcloud32: float32, pcloud64: float64}


def _get_cpp_dist(dist, pcloud_dtype):
    """Get the C++ weight function object, rebuilding at the target precision if needed."""
    target_float = _PCLOUD_TO_FLOAT[pcloud_dtype]
    cpp_obj = dist._cpp

    # Check if the stored C++ object already matches the target precision
    if target_float == float32:
        expected_types = (cpp.Gaussian32, cpp.Uniform32, cpp.Mixture32)
    else:
        expected_types = (cpp.Gaussian64, cpp.Uniform64, cpp.Mixture64)

    if isinstance(cpp_obj, expected_types):
        return cpp_obj

    # Precision mismatch — rebuild at the target precision
    if isinstance(dist, Gaussian):
        cls = {float32: cpp.Gaussian32, float64: cpp.Gaussian64}[target_float]
        return cls(dist.mean, dist.sigma)
    elif isinstance(dist, Uniform):
        cls = {float32: cpp.Uniform32, float64: cpp.Uniform64}[target_float]
        return cls(dist.lo, dist.hi)
    elif isinstance(dist, Mixture):
        mix_cls = {float32: cpp.Mixture32, float64: cpp.Mixture64}[target_float]
        cpp_components = [_get_cpp_dist(c, pcloud_dtype) for c in dist.components]
        return mix_cls(cpp_components, dist.weights)
    else:
        raise TypeError(f"Unsupported weight function type: {type(dist)}")


class IndexedPointCloudTensor(PointCloudTensor):
    """A PointCloudTensor backed by a shared source cloud + index arrays.

    Behaves identically to a regular PointCloudTensor. The indexed backing
    is an internal optimization — users interact with the same API.
    """

    __slots__ = ('_collection',)

    def __init__(self, collection):
        # Skip PointCloudTensor.__init__ — we don't have a C++ tensor of point clouds.
        # pylint: disable=super-init-not-called
        self._collection = collection
        self.dtype = (pcloud32
                      if isinstance(collection, cpp.IndexedPointCloudCollection32)
                      else pcloud64)

    @property
    def shape(self):
        return (self._collection.n_vantage(),)

    @property
    def ndim(self):
        return 1

    def __len__(self):
        return self._collection.n_vantage()

    def __getitem__(self, i):
        if isinstance(i, int):
            if i < 0:
                i += len(self)
            ipc = self._collection[i]
            return FloatTensor(ipc.materialize())
        raise TypeError(f"IndexedPointCloudTensor only supports integer indexing, got {type(i)}")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def indices(self):
        """Index tensor of shape (M, k) — source indices per vantage."""
        return self._collection.indices

    @property
    def source(self):
        """The shared source point cloud."""
        return FloatTensor(self._collection.source)


_DEFAULT_STAGES = (0.0, 0.1, 0.5, 1.0)
_DEFAULT_ESCALATION_THRESHOLD = 100
_DEFAULT_MAX_ATTEMPTS = 1000


class SamplingDiagnostics:
    """Post-hoc diagnostics for a sampling operation."""

    __slots__ = ('_cpp',)

    def __init__(self, cpp_diag):
        self._cpp = cpp_diag

    @property
    def acceptance_rate(self):
        """Per-vantage acceptance rate."""
        return self._cpp.acceptance_rate

    @property
    def total_attempts(self):
        """Per-vantage total number of proposal attempts."""
        return self._cpp.total_attempts

    @property
    def biased(self):
        """Per-vantage flag: True if adaptive escalation was triggered."""
        return self._cpp.biased

    @property
    def all_exact(self):
        """True if every vantage point was sampled without escalation."""
        return self._cpp.all_exact()

    @property
    def biased_vantage_count(self):
        """Number of vantage points where adaptive escalation was triggered."""
        return self._cpp.biased_vantage_count()


class SamplingResult:
    """Result of a sampling operation: samples plus diagnostics.

    Delegates to :class:`IndexedPointCloudTensor` for backward compatibility:
    ``.indices``, ``.source``, ``[i]``, ``len()`` all work as before.
    """

    __slots__ = ('_samples', '_diagnostics', 'dtype')

    def __init__(self, cpp_result, pcloud_dtype):
        self._samples = IndexedPointCloudTensor(cpp_result.collection)
        self._diagnostics = SamplingDiagnostics(cpp_result.diagnostics)
        self.dtype = pcloud_dtype

    @property
    def samples(self):
        """The sampled point clouds."""
        return self._samples

    @property
    def diagnostics(self):
        """Post-hoc sampling diagnostics."""
        return self._diagnostics

    # Delegate IndexedPointCloudTensor interface for backward compatibility

    @property
    def shape(self):
        return self._samples.shape

    @property
    def ndim(self):
        return self._samples.ndim

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i):
        return self._samples[i]

    def __iter__(self):
        return iter(self._samples)

    @property
    def indices(self):
        """Index tensor of shape (M, k) — source indices per vantage."""
        return self._samples.indices

    @property
    def source(self):
        """The shared source point cloud."""
        return self._samples.source


class DistanceWeightedSampler:
    """Precomputed sampling state for distance-weighted point cloud sampling.

    Builds a KD-tree over the source cloud once, then supports efficient
    repeated sampling with different vantage points and/or weight functions.

    Parameters
    ----------
    source : PointCloudTensor
        Source point cloud of shape ``(N, D)``. Must be a rank-1 tensor
        with one element.
    dtype : type, optional
        ``pcloud32`` or ``pcloud64``. Inferred from source if not given.
    """

    __slots__ = ('_cpp', '_dtype')

    def __init__(self, source, dtype=None):
        if dtype is None:
            dtype = source.dtype
        dtype = _validate_dtype(dtype, [pcloud32, pcloud64])
        self._dtype = dtype

        if isinstance(source, PointCloudTensor):
            source_inner = source._data._get_element([0])
        else:
            source_inner = source

        cpp_cls = {pcloud32: cpp.DistanceWeightedSampler32,
                   pcloud64: cpp.DistanceWeightedSampler64}[dtype]
        self._cpp = cpp_cls(source_inner)

    def sample(self, vantage, k, *, dist, radius=None, replace=True,
               generator=None, stages=_DEFAULT_STAGES,
               escalation_threshold=_DEFAULT_ESCALATION_THRESHOLD,
               max_attempts=_DEFAULT_MAX_ATTEMPTS):
        r"""Sample k points around each vantage point, weighted by a weight function of distance.

        Uses adaptive escalation: starts with exact (unbiased) sampling and
        automatically introduces bounded bias if acceptance rates are too low.
        Check :attr:`SamplingResult.diagnostics` to assess sampling quality.

        Parameters
        ----------
        vantage : PointCloudTensor
            Vantage points of shape ``(M, D)``.
        k : int
            Number of samples per vantage point.
        dist : Gaussian, Uniform, or Mixture
            Weight function applied to the Euclidean distance from each source
            point to the vantage point.
        radius : float, optional
            Ball radius restriction. If ``None``, samples from all points.
        replace : bool, optional
            If ``True`` (default), sample with replacement.
        generator : Generator, optional
            Random number generator. If ``None``, the global generator is used.
        stages : tuple of float, optional
            Correction-clamping escalation stages. Each value is a floor on the
            cumulative correction factor (0 = exact). Default: ``(0, 0.1, 0.5, 1.0)``.
        escalation_threshold : int, optional
            Consecutive rejections before escalating to the next stage.
            Default: 100.
        max_attempts : int, optional
            Maximum proposal attempts per requested sample. Default: 1000.

        Returns
        -------
        SamplingResult
            Contains the sampled point clouds and post-hoc diagnostics.
        """
        if isinstance(vantage, PointCloudTensor):
            vantage_inner = vantage._data._get_element([0])
        else:
            vantage_inner = vantage

        cpp_dist = _get_cpp_dist(dist, self._dtype)
        gen = generator._gen if generator is not None else None

        kwargs = dict(
            vantage=vantage_inner,
            k=k,
            dist=cpp_dist,
            replace=replace,
            generator=gen,
            stages=list(stages),
            escalation_threshold=int(escalation_threshold),
            max_attempts=int(max_attempts),
        )

        if radius is not None:
            kwargs['radius'] = float(radius)

        cpp_result = self._cpp.sample(**kwargs)
        return SamplingResult(cpp_result, self._dtype)

    @property
    def dim(self):
        """Dimensionality of the source point cloud."""
        return self._cpp.dim

    @property
    def n_points(self):
        """Number of points in the source cloud."""
        return self._cpp.n_points


def sample(source, vantage, k, *, dist, radius=None, replace=True,
           generator=None, dtype=None, stages=_DEFAULT_STAGES,
           escalation_threshold=_DEFAULT_ESCALATION_THRESHOLD,
           max_attempts=_DEFAULT_MAX_ATTEMPTS):
    r"""Sample k points around each vantage point, weighted by a weight function of distance.

    Convenience function that builds a :class:`DistanceWeightedSampler` from
    the source cloud and immediately samples. If you need to sample from the
    same source cloud multiple times, construct a
    :class:`DistanceWeightedSampler` directly to avoid rebuilding the KD-tree.

    Parameters
    ----------
    source : PointCloudTensor
        Source point cloud of shape ``(N, D)``. Must be a single point cloud
        (rank-1 tensor with one element), or the inner point cloud directly.
    vantage : PointCloudTensor
        Vantage points of shape ``(M, D)``.
    k : int
        Number of samples per vantage point.
    dist : Gaussian, Uniform, or Mixture
        Weight function applied to the Euclidean distance from each source
        point to the vantage point.
    radius : float, optional
        Ball radius restriction. If ``None``, samples from all points
        (tree nodes are not pruned by distance).
    replace : bool, optional
        If ``True`` (default), sample with replacement.
    generator : Generator, optional
        Random number generator. If ``None``, the global generator is used.
    dtype : type, optional
        ``pcloud32`` or ``pcloud64``. Inferred from source if not given.
    stages : tuple of float, optional
        Correction-clamping escalation stages. Default: ``(0, 0.1, 0.5, 1.0)``.
    escalation_threshold : int, optional
        Consecutive rejections before escalating. Default: 100.
    max_attempts : int, optional
        Maximum proposal attempts per requested sample. Default: 1000.

    Returns
    -------
    SamplingResult
        Contains the sampled point clouds and post-hoc diagnostics.
    """
    sampler = DistanceWeightedSampler(source, dtype=dtype)
    return sampler.sample(vantage, k, dist=dist, radius=radius, replace=replace,
                          generator=generator, stages=stages,
                          escalation_threshold=escalation_threshold,
                          max_attempts=max_attempts)
