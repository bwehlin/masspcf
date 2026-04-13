"""Stable rank transformer for sklearn pipelines."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._tags import InputTags

from ..persistence import barcode_to_stable_rank


class StableRank(BaseEstimator, TransformerMixin):
    """Sklearn transformer wrapping :func:`~masspcf.barcode_to_stable_rank`.

    Converts barcodes to stable rank PCFs.  To select a single homology
    dimension from the output, add a :class:`SelectDim` step after this
    one in the pipeline.

    Parameters
    ----------
    verbose : bool, optional
        Show progress information. Default False.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return barcode_to_stable_rank(X, verbose=self.verbose)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.input_tags = InputTags(two_d_array=False)
        return tags
