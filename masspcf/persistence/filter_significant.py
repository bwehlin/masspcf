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
from ..async_task import _run_task
from ..tensor import Tensor, _get_backend
from ..tensor_create import zeros
from ..typing import barcode32, barcode64
from .barcode import Barcode
from .ph_tensor import BarcodeTensor

cpp_p = cpp.persistence

_BACKEND_MAP = {
    barcode32: cpp_p.PersistenceBarcodeSummary32,
    barcode64: cpp_p.PersistenceBarcodeSummary64,
}


def filter_significant_bars(
    bc: Barcode | BarcodeTensor,
    alpha: float = 0.05,
    verbose: bool = False,
) -> Barcode | BarcodeTensor:
    r"""Filter a barcode to retain only statistically significant bars.

    Uses the universal null-distribution hypothesis test from
    Bobrowski & Skraba (2023) to distinguish signal from noise in
    persistence diagrams. Each bar is assigned a p-value based on its
    multiplicative persistence (death/birth ratio), using the universal
    Left-Gumbel null-distribution with Bonferroni correction.

    Bars with birth :math:`\leq 0` or infinite death are always retained.

    Parameters
    ----------
    bc : Barcode or BarcodeTensor
        A single barcode or a tensor of barcodes.
    alpha : float, optional
        Significance level for the hypothesis test (default 0.05).
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    Barcode or BarcodeTensor
        A single ``Barcode`` if the input is a single ``Barcode``,
        otherwise a ``BarcodeTensor`` with the same shape as the input.
        Each barcode contains only the statistically significant bars.

    References
    ----------
    Bobrowski, O. & Skraba, P. (2023). A universal null-distribution
    for topological data analysis. *Scientific Reports*, 13, 12274.
    """
    backend, X = _get_backend(bc, _BACKEND_MAP)

    if isinstance(X, Barcode):
        return Barcode(backend.filter_significant_bars(X._data, alpha))
    elif isinstance(X, Tensor):
        out = zeros((1,), dtype=X.dtype)

        _run_task(
            lambda: backend.spawn_filter_significant_task(
                X._data, out._data, alpha
            ),
            verbose=verbose,
        )

        if len(out.shape) == 2 and out.shape[0] == 1:
            out = out[0, :]

        return out
