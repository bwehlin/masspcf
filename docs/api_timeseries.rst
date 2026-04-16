masspcf.timeseries
==================

Time series with real-world time metadata.

.. automodule:: masspcf.timeseries
   :no-members:

Classes
-------

.. autoclass:: masspcf.timeseries.TimeSeries
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: masspcf.timeseries.TimeSeriesTensor
   :members:
   :undoc-members:
   :show-inheritance:

Interpolation strategies
------------------------

Built-in modes (``'nearest'`` and ``'linear'``) are selected by passing
the corresponding string to ``TimeSeries(..., interpolation=...)``. For
custom interpolation logic, wrap a Python callable in
:py:class:`CallableInterpolation`.

.. autoclass:: masspcf.timeseries.InterpolationStrategy
   :members:
   :show-inheritance:

.. autoclass:: masspcf.timeseries.CallableInterpolation
   :members:
   :show-inheritance:

Functions
---------

.. autofunction:: masspcf.timeseries.embed_time_delay
