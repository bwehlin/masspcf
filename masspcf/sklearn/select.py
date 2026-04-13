"""Index selection transformer for sklearn pipelines."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._tags import InputTags


class Select(BaseEstimator, TransformerMixin):
    """Apply an index expression to a tensor.

    Equivalent to ``X[index]``.  Useful for picking a homology
    dimension, slicing axes, or any other indexing operation in a
    pipeline.

    Parameters
    ----------
    index : tuple
        Index expression to apply, e.g. ``(..., 1)`` to select index 1
        along the last axis.

    Examples
    --------
    Extract H1 stable ranks from a ``(n_instances, 2)`` tensor::

        Select((..., 1))
    """

    def __init__(self, index):
        self.index = index

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.index]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.input_tags = InputTags(two_d_array=False)
        return tags
