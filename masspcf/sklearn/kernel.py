"""L2 kernel transformer for sklearn pipelines."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._tags import InputTags

from ..inner_product import l2_kernel


class PcfKernelTransformer(BaseEstimator, TransformerMixin):
    """Sklearn transformer wrapping :func:`~masspcf.l2_kernel`.

    Computes the :math:`L_2` kernel matrix for use with estimators that
    accept ``kernel="precomputed"`` (e.g.
    :class:`~sklearn.svm.SVC`).

    During ``fit``, the training data is stored. During ``transform``,
    the cross-kernel between the new data and the training data is
    returned.  ``fit_transform`` is optimized to compute the symmetric
    self-kernel directly.

    Parameters
    ----------
    verbose : bool, optional
        Show progress information. Default False.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, X, y=None):
        self.X_fit_ = X
        return self

    def transform(self, X):
        return np.asarray(
            l2_kernel(X, self.X_fit_, verbose=self.verbose))

    def fit_transform(self, X, y=None):
        self.X_fit_ = X
        return l2_kernel(X, verbose=self.verbose).to_dense()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.input_tags = InputTags(two_d_array=False, pairwise=True)
        return tags
