"""Scikit-learn compatible transformers for masspcf.

Requires scikit-learn to be installed::

    pip install scikit-learn
"""

try:
    import sklearn as _sklearn  # noqa: F401
except ImportError as e:
    raise ImportError(
        "masspcf.sklearn requires scikit-learn. "
        "Install it with: pip install scikit-learn"
    ) from e

from .embedding import TimeDelayEmbedding
from .homology import PersistentHomology
from .kernel import PcfKernelTransformer
from .stable_rank import StableRank

__all__ = [
    "TimeDelayEmbedding",
    "PersistentHomology",
    "StableRank",
    "PcfKernelTransformer",
]
