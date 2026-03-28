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

from . import _mpcf_cpp as cpp
from .async_task import _run_task
from .distance_matrix import DistanceMatrix
from .tensor import FloatTensor, PcfContainerLike, _get_backend, _to_tensor_pcf
from .typing import pcf32, pcf64


def _get_distance_backend(fs) -> type[cpp.Distance_f32_f32] | type[cpp.Distance_f64_f64]:
    mapping = {pcf32: cpp.Distance_f32_f32, pcf64: cpp.Distance_f64_f64}
    backend, _ = _get_backend(fs, mapping)
    return backend


def pdist(fs: PcfContainerLike, p=1, verbose=True) -> DistanceMatrix:
    r"""Compute the pairwise :math:`L_p` distance matrix for a 1-D tensor of PCFs.

    For a tensor :math:`(f_0, f_1, \ldots, f_{n-1})`, returns an
    :math:`n \times n` matrix :math:`D` where

    .. math::
        D_{ij} = \Vert f_i - f_j \Vert_p.

    Parameters
    ----------
    fs : PcfContainerLike
        A 1-D tensor of PCFs.
    p : float, optional
        The :math:`p` parameter in the :math:`L_p` distance (must be
        :math:`\geq 1`), by default 1.
    verbose : bool, optional
        Show progress information during computation, by default True.

    Returns
    -------
    DistanceMatrix
        A compressed symmetric distance matrix.

    Raises
    ------
    ValueError
        If ``fs`` is not 1-dimensional.
    """
    if p < 1:
        raise ValueError("p must be >= 1.")

    X = _to_tensor_pcf(fs)

    if len(X.shape) != 1:
        raise ValueError("1d tensor expected.")

    backend = _get_distance_backend(fs)

    if p == 1:
        task, dm_or_dense = backend.pdist_l1(X._data)
    else:
        task, dm_or_dense = backend.pdist_lp(X._data, float(p))

    _run_task(lambda: task, verbose=verbose)

    return DistanceMatrix(dm_or_dense)


def cdist(X: PcfContainerLike, Y: PcfContainerLike, p=1, verbose=True) -> FloatTensor:
    r"""Compute the pairwise :math:`L_p` distances between two tensors of PCFs.

    For tensors :math:`X` of shape :math:`(m_1, \ldots, m_n)` and :math:`Y` of
    shape :math:`(k_1, \ldots, k_l)`, returns a tensor of shape
    :math:`(m_1, \ldots, m_n, k_1, \ldots, k_l)` where

    .. math::
        D_{i_1, \ldots, i_n, j_1, \ldots, j_l}
        = \Vert X_{i_1, \ldots, i_n} - Y_{j_1, \ldots, j_l} \Vert_p.

    Parameters
    ----------
    X : PcfContainerLike
        A tensor of PCFs (any shape).
    Y : PcfContainerLike
        A tensor of PCFs (any shape, same dtype as X).
    p : float, optional
        The :math:`p` parameter in the :math:`L_p` distance (must be
        :math:`\geq 1`), by default 1.
    verbose : bool, optional
        Show progress information during computation, by default True.

    Returns
    -------
    FloatTensor
        A tensor of shape ``(*X.shape, *Y.shape)`` containing pairwise distances.
    """
    if p < 1:
        raise ValueError("p must be >= 1.")

    Xt = _to_tensor_pcf(X)
    Yt = _to_tensor_pcf(Y)

    backend = _get_distance_backend(X)

    if p == 1:
        task, out = backend.cdist_l1(Xt._data, Yt._data)
    else:
        task, out = backend.cdist_lp(Xt._data, Yt._data, float(p))

    _run_task(lambda: task, verbose=verbose)

    return FloatTensor(out)
