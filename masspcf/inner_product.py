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
from .symmetric_matrix import SymmetricMatrix
from .tensor import FloatTensor, PcfContainerLike, _resolve_pcf_inputs
from .typing import pcf32, pcf64


_INNER_PRODUCT_BACKEND_MAP = {pcf32: cpp.InnerProduct_f32_f32, pcf64: cpp.InnerProduct_f64_f64}


def l2_kernel(X: PcfContainerLike, Y: PcfContainerLike = None, verbose=False):
    r"""Compute the :math:`L_2` kernel matrix for one or two tensors of PCFs.

    **Single tensor** (``Y`` omitted): returns the symmetric kernel matrix

    .. math::
        K_{ij} = \int_0^\infty f_i(t) \, f_j(t) \, dt.

    **Two tensors**: returns the cross-kernel matrix

    .. math::
        K_{ij} = \int_0^\infty X_i(t) \, Y_j(t) \, dt

    with shape ``(*X.shape, *Y.shape)``.

    Parameters
    ----------
    X : PcfContainerLike
        A 1-D tensor of PCFs.
    Y : PcfContainerLike, optional
        A second tensor of PCFs. When provided, the cross-kernel is
        computed instead of the self-kernel.
    verbose : bool, optional
        Show progress information during computation, by default False.

    Returns
    -------
    SymmetricMatrix or FloatTensor
        A ``SymmetricMatrix`` when ``Y`` is omitted, or a ``FloatTensor``
        of shape ``(*X.shape, *Y.shape)`` when ``Y`` is provided.
    """
    if Y is None:
        backend, Xt = _resolve_pcf_inputs(_INNER_PRODUCT_BACKEND_MAP, X)

        if len(Xt.shape) != 1:
            raise ValueError("1d tensor expected.")
        task, sm_or_dense = backend.l2(Xt._data)
        _run_task(lambda: task, verbose=verbose)

        if isinstance(sm_or_dense, np.ndarray):
            return SymmetricMatrix.from_dense(sm_or_dense)
        else:
            return SymmetricMatrix(sm_or_dense)
    else:
        backend, Xt, Yt = _resolve_pcf_inputs(
            _INNER_PRODUCT_BACKEND_MAP, X, Y)

        task, out = backend.l2_cross(Xt._data, Yt._data)
        _run_task(lambda: task, verbose=verbose)

        return FloatTensor(out)
