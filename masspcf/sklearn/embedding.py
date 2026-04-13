"""Time delay embedding transformer for sklearn pipelines."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._tags import Tags, InputTags

from ..timeseries import TimeSeries, TimeSeriesTensor, embed_time_delay


class TimeDelayEmbedding(BaseEstimator, TransformerMixin):
    """Sklearn transformer wrapping :func:`~masspcf.embed_time_delay`.

    Accepts either a ``TimeSeriesTensor`` or a raw NumPy array of shape
    ``(n_instances, n_times, n_channels)`` (channels-last).  In the
    latter case, each instance is converted to a
    :class:`~masspcf.TimeSeries` with the given ``time_step``.

    Parameters
    ----------
    dimension : int
        Embedding dimension.
    delay : float
        Time delay between components of the embedding vector.
    time_step : float, optional
        Sampling interval, used when the input is a raw NumPy array.
        Default 1.0.
    window : float or None, optional
        Window duration for splitting into segments. ``None`` produces a
        single point cloud per instance.
    stride : float or None, optional
        Step between window starts. Defaults to ``window``
        (non-overlapping).
    """

    def __init__(self, dimension, delay, time_step=1.0, window=None,
                 stride=None):
        self.dimension = dimension
        self.delay = delay
        self.time_step = time_step
        self.window = window
        self.stride = stride

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = TimeSeriesTensor([
                TimeSeries(X[i], start_time=0.0, time_step=self.time_step)
                for i in range(len(X))
            ])
        return embed_time_delay(
            X,
            dimension=self.dimension,
            delay=self.delay,
            window=self.window,
            stride=self.stride,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.input_tags = InputTags(two_d_array=False)
        return tags
