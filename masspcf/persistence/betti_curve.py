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


def barcode_to_betti_curve(
    bc: Barcode | Barcode32Tensor | Barcode64Tensor, verbose=True
):
    r"""Convert barcodes to Betti curves.

    The Betti curve is the PCF that counts, for each filtration value
    :math:`t \geq 0`, the number of bars alive at :math:`t`
    (i.e., bars with birth :math:`\leq t <` death).

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
    """

    backend, X = _get_backend(
        bc,
        {
            barcode32: cpp_p.PersistenceBettiCurve32,
            barcode64: cpp_p.PersistenceBettiCurve64,
        },
    )

    if isinstance(X, Barcode):
        return Pcf(backend.barcode_to_betti_curve(X._data))
    elif isinstance(X, Tensor):
        if isinstance(X, Barcode32Tensor):
            out = zeros((1,), dtype=pcf32)
        elif isinstance(X, Barcode64Tensor):
            out = zeros((1,), dtype=pcf64)

        task = None
        try:
            task = backend.spawn_barcode_to_betti_curve_task(X._data, out._data)
            _wait_for_task(task, verbose)
        finally:
            if task is not None:
                task.request_stop()
                _wait_for_task(task, verbose=verbose)

        if len(out.shape) == 2 and out.shape[0] == 1:
            out = out[0, :]

        return out
