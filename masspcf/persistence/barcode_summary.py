#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from .. import _mpcf_cpp as cpp
from ..async_task import _wait_for_task
from ..pcf import Pcf
from ..tensor import Tensor, _get_backend
from ..tensor_create import zeros
from ..typing import barcode32, barcode64, pcf32, pcf64
from .barcode import Barcode
from .ph_tensor import Barcode32Tensor, Barcode64Tensor

cpp_p = cpp.persistence

_BACKEND_MAP = {
    barcode32: cpp_p.PersistenceBarcodeSummary32,
    barcode64: cpp_p.PersistenceBarcodeSummary64,
}


def _barcode_to_pcf(bc, single_method, task_method, verbose=True):
    """Shared implementation for barcode-to-PCF conversions."""

    backend, X = _get_backend(bc, _BACKEND_MAP)

    if isinstance(X, Barcode):
        return Pcf(getattr(backend, single_method)(X._data))
    elif isinstance(X, Tensor):
        if isinstance(X, Barcode32Tensor):
            out = zeros((1,), dtype=pcf32)
        elif isinstance(X, Barcode64Tensor):
            out = zeros((1,), dtype=pcf64)

        task = None
        try:
            task = getattr(backend, task_method)(X._data, out._data)
            _wait_for_task(task, verbose)
        finally:
            if task is not None:
                task.request_stop()
                _wait_for_task(task, verbose=verbose)

        if len(out.shape) == 2 and out.shape[0] == 1:
            out = out[0, :]

        return out


def barcode_to_stable_rank(
    bc: Barcode | Barcode32Tensor | Barcode64Tensor, verbose=True
):
    r"""Convert barcodes to stable rank functions.

    The stable rank of a barcode is the PCF that counts, for each
    :math:`t \geq 0`, the number of bars with length (death minus birth)
    strictly greater than :math:`t` [1]_.

    Parameters
    ----------
    bc : Barcode, Barcode32Tensor, or Barcode64Tensor
        A single barcode or a tensor of barcodes.
    verbose : bool, optional
        Show progress information, by default True.

    Returns
    -------
    Pcf or PcfTensor
        A single ``Pcf`` if the input is a single ``Barcode``, otherwise a
        ``PcfTensor`` with the same shape as the input.

    References
    ----------
    .. [1] W. Chachólski and H. Riihimäki, "Metrics and stabilization in
       one parameter persistence", *SIAM Journal on Applied Algebra and
       Geometry*, vol. 4, no. 1, pp. 69--98, 2020.
    """
    return _barcode_to_pcf(
        bc, "barcode_to_stable_rank", "spawn_barcode_to_stable_rank_task", verbose
    )


def barcode_to_betti_curve(
    bc: Barcode | Barcode32Tensor | Barcode64Tensor, verbose=True
):
    r"""Convert barcodes to Betti curves.

    The Betti curve is the PCF that counts, for each filtration value
    :math:`t \geq 0`, the number of bars alive at :math:`t`
    (i.e., bars with birth :math:`\leq t <` death) [1]_ [2]_.

    Parameters
    ----------
    bc : Barcode, Barcode32Tensor, or Barcode64Tensor
        A single barcode or a tensor of barcodes.
    verbose : bool, optional
        Show progress information, by default True.

    Returns
    -------
    Pcf or PcfTensor
        A single ``Pcf`` if the input is a single ``Barcode``, otherwise a
        ``PcfTensor`` with the same shape as the input.

    References
    ----------
    .. [1] Y. Umeda, "Time series classification via topological data
       analysis", *Information and Media Technologies*, vol. 12,
       pp. 228--239, 2017.
    .. [2] F. Chazal and B. Michel, "An introduction to topological data
       analysis: fundamental and practical aspects for data scientists",
       *Frontiers in Artificial Intelligence*, vol. 4, 667963, 2021.
    """
    return _barcode_to_pcf(
        bc, "barcode_to_betti_curve", "spawn_barcode_to_betti_curve_task", verbose
    )
