"""Stable rank transformer for sklearn pipelines."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._tags import InputTags

from ..persistence import barcode_to_stable_rank


class StableRank(BaseEstimator, TransformerMixin):
    """Sklearn transformer wrapping :func:`~masspcf.barcode_to_stable_rank`.

    Converts barcodes to stable rank PCFs, with optional homology
    dimension selection.

    Parameters
    ----------
    dim : int or None, optional
        Homology dimension to extract from the last axis (e.g. ``1``
        for H1). ``None`` keeps all dimensions.
    verbose : bool, optional
        Show progress information. Default False.
    """

    def __init__(self, dim=None, verbose=False):
        self.dim = dim
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sranks = barcode_to_stable_rank(X, verbose=self.verbose)

        if self.dim is not None:
            # Select one homology dimension from the last axis
            # e.g. shape (80, 6, 2) -> sranks[:, :, 1] -> (80, 6)
            idx = tuple(
                slice(None) if i < len(sranks.shape) - 1 else self.dim
                for i in range(len(sranks.shape))
            )
            sranks = sranks[idx]

        return sranks

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.input_tags = InputTags(two_d_array=False)
        return tags
