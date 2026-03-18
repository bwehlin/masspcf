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

import numpy as np

from . import _mpcf_cpp as cpp
from .async_task import _run_task
from .np_support import numpy_type
from .tensor import PcfContainerLike, _get_backend, _to_tensor_pcf
from .typing import pcf32, pcf64


def _get_distance_backend(fs) -> cpp.Distance_f32_f32 | cpp.Distance_f64_f64:
    mapping = {pcf32: cpp.Distance_f32_f32, pcf64: cpp.Distance_f64_f64}

    return _get_backend(fs, mapping)


def pdist(fs: PcfContainerLike, p=1, verbose=True):
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
    numpy.ndarray
        An :math:`n \times n` distance matrix.

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

    backend, fs = _get_distance_backend(fs)
    matrix = np.zeros((X.shape[0], X.shape[0]), dtype=numpy_type(X))

    _run_task(lambda: backend.pdist_l1(matrix, X._data), verbose=verbose)
    return matrix
