"""Persistent homology transformer for sklearn pipelines."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._tags import InputTags

from ..persistence import compute_persistent_homology


class PersistentHomology(BaseEstimator, TransformerMixin):
    """Sklearn transformer wrapping :func:`~masspcf.compute_persistent_homology`.

    Parameters
    ----------
    max_dim : int, optional
        Maximum homology dimension to compute. Default 1.
    reduced : bool, optional
        If True, compute reduced homology. Default False.
    verbose : bool, optional
        Show progress information. Default False.
    """

    def __init__(self, max_dim=1, reduced=False, verbose=False):
        self.max_dim = max_dim
        self.reduced = reduced
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return compute_persistent_homology(
            X,
            max_dim=self.max_dim,
            reduced=self.reduced,
            verbose=self.verbose,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.input_tags = InputTags(two_d_array=False)
        return tags
