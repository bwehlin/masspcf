"""Reduction transformers for sklearn pipelines."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._tags import InputTags

from ..reductions import mean


class Mean(BaseEstimator, TransformerMixin):
    """Sklearn transformer that computes the pointwise mean of PCF tensors.

    Wraps :func:`~masspcf.mean` as a pipeline step, reducing a tensor
    along the specified dimension.

    Parameters
    ----------
    dim : int or None, optional
        Tensor axis to reduce. ``None`` defaults to the last axis.
    """

    def __init__(self, dim=None):
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dim = self.dim
        if dim is None:
            dim = len(X.shape) - 1
        return mean(X, dim=dim)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.input_tags = InputTags(two_d_array=False)
        return tags
