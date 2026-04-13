#    Copyright 2024-2026 Bjorn Wehlin
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import annotations

import numpy as np

from .. import _mpcf_cpp as cpp
from .._tensor_base import FunctionTensorMixin, Tensor, _tensor_from_nested
from ..functional.pcf import Pcf
from ..tensor import PointCloudTensor
from ..typing import (
    _assert_valid_dtype,
    pcloud32,
    pcloud64,
    ts32,
    ts64,
)


def _datetime_unit(dt_or_arr):
    """Extract the unit string from a datetime64/timedelta64 scalar or array."""
    return np.datetime_data(np.dtype(dt_or_arr.dtype))[0]


# Variable-length datetime units that cannot use the C++ chrono path directly.
# Actual calendar dates are computed via numpy and converted to exact float
# seconds (no approximation).
_VARIABLE_LENGTH_UNITS = {'M', 'Y'}


def _dt64_to_float_seconds(t):
    """Convert a datetime64 scalar to float seconds since Unix epoch.

    Uses the input's native unit to avoid artificial precision loss.
    For variable-length units (M, Y), casts to ``datetime64[s]`` first
    since numpy cannot compute a tick-to-seconds ratio for those units.
    The only limit is float64 (~15-16 significant digits).
    """
    unit = _datetime_unit(t)
    if unit in _VARIABLE_LENGTH_UNITS:
        t = t.astype('datetime64[s]')
        unit = 's'
    ticks = int(t.view('int64'))
    sec_per_tick = np.timedelta64(1, unit) / np.timedelta64(1, 's')
    return ticks * float(sec_per_tick)


def _dt64_array_to_float_seconds(arr):
    """Convert a datetime64 array to float64 seconds since Unix epoch.

    Uses the input's native unit to avoid artificial precision loss.
    For variable-length units (M, Y), casts to ``datetime64[s]`` first.
    """
    unit = _datetime_unit(arr)
    if unit in _VARIABLE_LENGTH_UNITS:
        arr = arr.astype('datetime64[s]')
        unit = 's'
    ticks = arr.view('int64').astype('float64')
    sec_per_tick = np.timedelta64(1, unit) / np.timedelta64(1, 's')
    return ticks * float(sec_per_tick)


class _DateTimeConverter:
    """Converts between numpy datetime64 and int64 ticks for the C++ chrono path.

    The C++ side stores start_time/time_step as float seconds. For
    fixed-length units (as through W), query times are passed as int64 ticks
    + unit string so that C++ can construct the appropriate
    ``std::chrono::duration`` and perform exact integer arithmetic before
    converting to float.

    For variable-length units (M, Y), numpy's calendar-aware arithmetic
    computes exact dates, which are then converted to float seconds.
    """

    def __init__(self, start_time, time_step):
        self.start_time_dt = start_time
        self.time_step_td = time_step
        self._unit_str = _datetime_unit(time_step)
        self._unit_td = np.timedelta64(1, self._unit_str)
        self._variable_length = self._unit_str in _VARIABLE_LENGTH_UNITS

    def start_ticks(self):
        """Return start_time as int64 ticks in the native unit."""
        return int(self.start_time_dt.astype(
            f"datetime64[{self._unit_str}]").view("int64"))

    def step_ticks(self):
        """Return time_step as int64 ticks in the native unit."""
        return int(self.time_step_td / self._unit_td)

    def array_to_int64(self, arr):
        """Convert a datetime64 array to int64 ticks."""
        return arr.astype(
            f"datetime64[{self._unit_str}]").view("int64").astype("int64")


def _dt64_to_ticks(t):
    """Convert a datetime64 scalar to (int64_ticks, unit_string).

    For variable-length units (M, Y), returns ``(float_seconds, None)``.
    """
    unit = _datetime_unit(t)
    if unit in _VARIABLE_LENGTH_UNITS:
        return _dt64_to_float_seconds(t), None
    ticks = int(t.astype(f"datetime64[{unit}]").view("int64"))
    return ticks, unit


def _dt64_array_to_ticks(arr):
    """Convert a datetime64 array to (int64_ticks_array, unit_string).

    For variable-length units (M, Y), returns ``(float_seconds_array, None)``.
    """
    unit = _datetime_unit(arr)
    if unit in _VARIABLE_LENGTH_UNITS:
        return _dt64_array_to_float_seconds(arr), None
    ticks = arr.astype(f"datetime64[{unit}]").view("int64").astype("int64")
    return ticks, unit


_TS_CPP_TO_DTYPE = {
    cpp.TimeSeries32Tensor: ts32,
    cpp.TimeSeries64Tensor: ts64,
}


_INTERP_STR_TO_CPP = {
    'nearest': cpp.InterpolationMode.nearest,
    'linear': cpp.InterpolationMode.linear,
}
_INTERP_CPP_TO_STR = {v: k for k, v in _INTERP_STR_TO_CPP.items()}


class TimeSeries:
    """A time series stored as a piecewise constant function with time metadata.

    Each ``TimeSeries`` wraps a :class:`~masspcf.Pcf` together with a
    ``start_time`` (the real-world time of the first sample) and a
    ``time_step`` (the duration between samples).

    Construction forms:

    ``TimeSeries(times, values)``
        Explicit time points. *times* is a 1-D array of sample times
        (float or ``datetime64``), *values* a 1-D array of sample values
        or a 2-D array of shape ``(n_times, n_channels)``.
        ``start_time`` is inferred from ``times[0]``.

    ``TimeSeries(values, *, start_time=0.0, time_step=1.0)``
        Regular sampling. *values* is a 1-D array or a 2-D array of
        shape ``(n_times, n_channels)``; sample *i* is placed
        at ``start_time + i * time_step``.

    ``TimeSeries(existing)``
        Copy an existing ``TimeSeries``.

    Parameters
    ----------
    start_time : float, int, or numpy.datetime64, optional
        Start time for regularly-sampled construction. Default 0.0.
        When ``datetime64``, times are stored internally as seconds
        since the Unix epoch. All numpy datetime64 units are supported
        (``as`` through ``Y``). For variable-length units (``M``, ``Y``),
        numpy's calendar-aware arithmetic computes exact dates, which
        are then converted to float seconds.
    time_step : float, int, or numpy.timedelta64, optional
        Spacing between samples for regularly-sampled construction.
        Default 1.0. When ``timedelta64``, converted to seconds.
        Not used when *times* is provided.
    dtype : masspcf.dtype, optional
        ``ts32`` or ``ts64``. If ``None``, inferred from data.
    interpolation : str, optional
        Interpolation mode: ``'nearest'`` (default) returns the value of
        the left breakpoint (piecewise constant), ``'linear'`` linearly
        interpolates between adjacent breakpoints.

    Notes
    -----
    **Datetime precision.** Times are stored internally as float64
    seconds. Float64 has about 15--16 significant digits, so precision
    degrades for times far from zero. Near the current epoch (~1.7e9 s),
    the smallest representable difference is roughly 200 ns. Near time
    zero, the full precision of the input unit is preserved.

    **Timezone handling.** All ``datetime64`` values must be
    timezone-naive. If you have timezone-aware timestamps, convert to
    UTC and strip the timezone before constructing a ``TimeSeries``.
    """

    _NP_TO_CPP_TS = {
        np.float32: cpp.TimeSeries_f32_f32,
        np.float64: cpp.TimeSeries_f64_f64,
    }

    _CPP_TO_DTYPE = {
        cpp.TimeSeries_f32_f32: ts32,
        cpp.TimeSeries_f64_f64: ts64,
    }

    _DTYPE_TO_NP = {
        ts32: np.float32,
        ts64: np.float64,
    }

    def __init__(self, times_or_values, values=None, *,
                 start_time=None, time_step=1.0, dtype=None,
                 interpolation='nearest'):
        if interpolation not in _INTERP_STR_TO_CPP:
            raise ValueError(
                f"interpolation must be 'nearest' or 'linear', "
                f"got {interpolation!r}")

        # --- Copy / C++ wrap ---
        if values is None and start_time is None:
            if isinstance(times_or_values, TimeSeries):
                self._data = times_or_values._data
                self._data.interpolation = _INTERP_STR_TO_CPP[interpolation]
                return
            if isinstance(times_or_values, tuple(self._CPP_TO_DTYPE.keys())):
                self._data = times_or_values
                return

        # Resolve dtype for values
        def _resolve_dtype(arr):
            if dtype is not None:
                _assert_valid_dtype(dtype, (ts32, ts64))
                return self._DTYPE_TO_NP[dtype]
            if arr.dtype.type in self._NP_TO_CPP_TS:
                return arr.dtype.type
            return np.float64

        if values is not None:
            # --- (times, values) form ---
            times = np.asarray(times_or_values)
            values = np.asarray(values)
            if times.ndim != 1:
                raise ValueError("times must be a 1-D array")
            if values.ndim not in (1, 2):
                raise ValueError("values must be 1-D or 2-D")
            if values.ndim == 1 and len(times) != len(values):
                raise ValueError(
                    "times and values must have the same length")
            if values.ndim == 2 and len(times) != values.shape[0]:
                raise ValueError(
                    "times length must match values first dimension")
            if len(times) < 2:
                raise ValueError(
                    "times must have at least 2 elements")

            np_dtype = _resolve_dtype(values)
            values = values.astype(np_dtype)
            cpp_cls = self._NP_TO_CPP_TS[np_dtype]

            if np.issubdtype(times.dtype, np.datetime64):
                inferred_step = times[1] - times[0]
                dt_converter = _DateTimeConverter(times[0], inferred_step)
                if dt_converter._variable_length:
                    times_s = _dt64_array_to_float_seconds(times).astype(
                        np_dtype)
                    self._data = cpp_cls(times_s, values)
                else:
                    ticks = dt_converter.array_to_int64(times)
                    self._data = cpp_cls(
                        ticks, values,
                        dt_converter.step_ticks(), dt_converter._unit_str)
            else:
                # Store real offsets from start as PCF breakpoints,
                # time_step=1 so evaluate is just pcf_t = query - start
                times_f = times.astype(np_dtype)
                self._data = cpp_cls(times_f, values)

        else:
            # --- (values, start_time=, time_step=) form ---
            vals = np.asarray(times_or_values)
            if vals.ndim not in (1, 2):
                raise ValueError("values must be 1-D or 2-D")

            np_dtype = _resolve_dtype(vals)
            vals = vals.astype(np_dtype)
            cpp_cls = self._NP_TO_CPP_TS[np_dtype]

            if start_time is None:
                start_time = 0.0

            n = vals.shape[0]

            if isinstance(start_time, np.datetime64):
                if not isinstance(time_step, np.timedelta64):
                    raise TypeError(
                        "time_step must be timedelta64 when start_time "
                        "is datetime64")
                if time_step <= np.timedelta64(0):
                    raise ValueError("time_step must be positive")
                dt_converter = _DateTimeConverter(start_time, time_step)
                if dt_converter._variable_length:
                    # Use numpy's calendar-aware arithmetic to compute
                    # exact dates, then convert to float seconds.
                    dates = start_time + np.arange(n) * time_step
                    times_arr = _dt64_array_to_float_seconds(dates).astype(
                        np_dtype)
                    self._data = cpp_cls(times_arr, vals)
                else:
                    self._data = cpp_cls(
                        vals, dt_converter.start_ticks(),
                        dt_converter.step_ticks(), dt_converter._unit_str)
            else:
                # Build explicit times: start + i * step
                start_f = float(start_time)
                step_f = float(time_step)
                if step_f <= 0:
                    raise ValueError("time_step must be positive")
                times_arr = (start_f + np.arange(n, dtype=np_dtype) * step_f)
                self._data = cpp_cls(times_arr, vals)

        self._data.interpolation = _INTERP_STR_TO_CPP[interpolation]

    def __call__(self, t, *, snap_tol=None):
        """Evaluate the time series at the given time(s).

        Parameters
        ----------
        t : float, int, numpy.datetime64, or numpy.ndarray
            Query time(s). If ``start_time`` is a datetime64, *t* should
            also be datetime64 (converted to PCF-internal time via the
            datetime converter for full precision).
        snap_tol : float or None, optional
            Relative error tolerance for snapping query times to
            breakpoints. When the relative difference between a query
            time and the nearest breakpoint is below this threshold,
            the query snaps to that breakpoint. Prevents floating-point
            rounding from returning the wrong interval. ``None`` uses
            the type-dependent default (``1e-9`` for float64, ``1e-5``
            for float32). Set to ``0`` to disable snapping.

        Returns
        -------
        float or numpy.ndarray
            Value(s) at the queried time(s). NaN for times outside the
            series domain.
        """
        kw = {} if snap_tol is None else {'snap_tol': snap_tol}
        if isinstance(t, np.datetime64):
            ticks, unit = _dt64_to_ticks(t)
            if unit is None:
                return self._data(float(ticks), **kw)
            return self._data(ticks, unit, **kw)
        if isinstance(t, np.ndarray) and np.issubdtype(t.dtype, np.datetime64):
            ticks, unit = _dt64_array_to_ticks(t)
            if unit is None:
                return self._data(ticks, **kw)
            return self._data(ticks, unit, **kw)
        if isinstance(t, np.ndarray):
            return self._data(t, **kw)
        if isinstance(t, (int, float)):
            return self._data(t, **kw)
        if isinstance(t, list):
            return self._data(np.asarray(t, dtype=np.float64), **kw)
        raise TypeError(f"Cannot evaluate TimeSeries at type {type(t)}")

    @property
    def dtype(self):
        """The dtype of this time series (``ts32`` or ``ts64``)."""
        return self._CPP_TO_DTYPE[type(self._data)]

    @property
    def start_time(self):
        """The real-world time of the first sample (seconds since Unix epoch
        when constructed from ``datetime64``)."""
        return self._data.start_time

    @property
    def end_time(self):
        """The real-world time of the last sample."""
        return self._data.end_time

    @property
    def interpolation(self):
        """Interpolation mode: ``'nearest'`` or ``'linear'``."""
        return _INTERP_CPP_TO_STR[self._data.interpolation]

    @interpolation.setter
    def interpolation(self, value):
        if value not in _INTERP_STR_TO_CPP:
            raise ValueError(
                f"interpolation must be 'nearest' or 'linear', "
                f"got {value!r}")
        self._data.interpolation = _INTERP_STR_TO_CPP[value]

    @property
    def n_channels(self):
        """Number of channels."""
        return self._data.n_channels

    @property
    def n_times(self):
        """Number of time points (breakpoints)."""
        return self._data.n_times

    @property
    def values(self):
        """Sample values. Shape ``(n_times,)`` for single-channel,
        ``(n_times, n_channels)`` for multi-channel."""
        nc = self._data.n_channels
        nt = self._data.n_times
        flat = self._data._values  # row-major: n_times * n_channels
        if nc == 1:
            return flat.copy()
        return flat.reshape(nt, nc).copy()

    @property
    def times(self):
        """The real-world times for each sample (seconds since Unix epoch
        when constructed from ``datetime64``)."""
        return (self._data.start_time
                + self._data._internal_times * self._data.time_step)

    def __repr__(self):
        nc = self.n_channels
        chan_str = f", n_channels={nc}" if nc > 1 else ""
        interp = self.interpolation
        interp_str = f", interpolation='{interp}'" if interp != 'nearest' else ""
        return (
            f"TimeSeries(start_time={self.start_time}, "
            f"n_times={self.n_times}{chan_str}, "
            f"dtype={self.dtype}{interp_str})"
        )

    def __len__(self):
        return self.n_times

    def __eq__(self, other):
        if not isinstance(other, TimeSeries):
            return NotImplemented
        return self._data == other._data

    def __ne__(self, other):
        if not isinstance(other, TimeSeries):
            return NotImplemented
        return self._data != other._data

    def __reduce__(self):
        import io as _io
        from ..io import _save_object, _unpickle_object
        buf = _io.BytesIO()
        _save_object(self, buf)
        return _unpickle_object, (buf.getvalue(),)


class TimeSeriesTensor(Tensor, FunctionTensorMixin):
    """A tensor of :class:`TimeSeries` objects.

    Parameters
    ----------
    data : list, tuple, or C++ TimeSeries tensor
        Input data.

        * **list/tuple of TimeSeries**: builds a tensor from the list.
        * **C++ tensor** (``TimeSeries32Tensor`` / ``TimeSeries64Tensor``):
          wraps directly.
    """

    def __init__(self, data):
        super().__init__()
        if isinstance(data, TimeSeriesTensor):
            data = data._data
        elif isinstance(data, (list, tuple)):
            data = _tensor_from_nested(data, {
                cpp.TimeSeries_f32_f32: cpp.TimeSeries32Tensor,
                cpp.TimeSeries_f64_f64: cpp.TimeSeries64Tensor,
            })
        elif not isinstance(data, (cpp.TimeSeries32Tensor, cpp.TimeSeries64Tensor)):
            raise TypeError(f"Cannot create TimeSeriesTensor from {type(data)}")
        self._data = data

    @property
    def dtype(self):
        """The dtype of this tensor (``ts32`` or ``ts64``)."""
        return _TS_CPP_TO_DTYPE[type(self._data)]

    def __call__(self, t, *, snap_tol=None):
        """Evaluate every series at the given time(s).

        Parameters
        ----------
        t : float, int, numpy.datetime64, or numpy.ndarray
            Query time(s).
        snap_tol : float or None, optional
            Relative error tolerance for breakpoint snapping.
            See :meth:`TimeSeries.__call__` for details.

        Supports ``datetime64`` scalars and arrays. Datetime values are
        passed as int64 ticks + unit string to C++, where each element's
        chrono evaluate converts to seconds independently.
        """
        kw = {} if snap_tol is None else {'snap_tol': snap_tol}
        if isinstance(t, np.datetime64):
            ticks, unit = _dt64_to_ticks(t)
            if unit is None:
                return np.asarray(self._data(float(ticks), **kw))
            return np.asarray(self._data(ticks, unit, **kw))
        if isinstance(t, np.ndarray) and np.issubdtype(t.dtype, np.datetime64):
            ticks, unit = _dt64_array_to_ticks(t)
            if unit is None:
                return np.asarray(self._data(ticks, **kw))
            return np.asarray(self._data(ticks, unit, **kw))
        if snap_tol is not None:
            return np.asarray(self._data(t, snap_tol=snap_tol))
        return super().__call__(t)

    def _to_py_tensor(self, data):
        return TimeSeriesTensor(data)

    def _represent_element(self, element):
        return TimeSeries(element)

    def _decay_value(self, val):
        return val._data

    def _get_valid_setitem_dtypes(self):
        return [TimeSeriesTensor, TimeSeries]

    @property
    def start_times(self):
        """1-D array of start times for each series."""
        flat = self._data.flatten()
        n = flat.shape[0]
        return np.array([flat._get_element([i]).start_time for i in range(n)])

    @property
    def end_times(self):
        """1-D array of end times for each series."""
        flat = self._data.flatten()
        n = flat.shape[0]
        return np.array([flat._get_element([i]).end_time for i in range(n)])


_TS_DTYPE_TO_EMBED = {
    ts32: (cpp.embed_time_delay_f32, cpp.embed_time_delay_tensor_f32, pcloud32),
    ts64: (cpp.embed_time_delay_f64, cpp.embed_time_delay_tensor_f64, pcloud64),
}


def embed_time_delay(ts, dimension, delay, *, window=None, stride=None,
                     snap_tol=None):
    """Compute time delay embedding of a time series.

    At each valid time *t*, the embedding vector looks backward::

        [x(t-(d-1)*tau), ..., x(t-tau), x(t)]

    where *d* is the embedding ``dimension`` and *tau* is the ``delay``.

    Parameters
    ----------
    ts : TimeSeries or TimeSeriesTensor
        Input time series.
    dimension : int
        Embedding dimension (>= 1).
    delay : float or numpy.timedelta64
        Real-time delay between embedding components.
    window : float, numpy.timedelta64, or None
        If given, split the valid time range into windows of this duration.
        Windows are anchored at the end of the range and extend backward,
        so the first window may be shorter than *window*.
    stride : float, numpy.timedelta64, or None
        Step between window starts. Defaults to *window* (non-overlapping).
    snap_tol : float or None, optional
        Relative error tolerance for snapping evaluation times to
        breakpoints. See :meth:`TimeSeries.__call__` for details.
        ``None`` uses the type-dependent default.

    Returns
    -------
    PointCloudTensor
        For a single ``TimeSeries`` without windowing, shape ``(1,)``
        containing one point cloud.  With windowing, shape
        ``(n_windows,)``.  For a ``TimeSeriesTensor`` of shape *S*,
        shape *S* (no windowing) or *S* + ``(n_windows,)`` (windowed).
        Each point cloud has shape ``(n_points, dimension * n_channels)``.
    """
    if dimension < 1:
        raise ValueError("dimension must be >= 1")

    def _to_float(val):
        if val is None:
            return 0.0
        if isinstance(val, np.timedelta64):
            return val / np.timedelta64(1, 's')
        return float(val)

    delay_f = _to_float(delay)
    if delay_f <= 0:
        raise ValueError("delay must be positive")
    window_f = _to_float(window)
    stride_f = _to_float(stride)

    kw = {} if snap_tol is None else {'snap_tol': snap_tol}

    if isinstance(ts, TimeSeries):
        embed_fn, _, _ = _TS_DTYPE_TO_EMBED[ts.dtype]
        result = embed_fn(ts._data, dimension, delay_f, window_f, stride_f,
                          **kw)
        return PointCloudTensor(result)
    elif isinstance(ts, TimeSeriesTensor):
        _, embed_tensor_fn, _ = _TS_DTYPE_TO_EMBED[ts.dtype]
        result = embed_tensor_fn(
            ts._data, dimension, delay_f, window_f, stride_f, **kw)
        return PointCloudTensor(result)
    else:
        raise TypeError(
            f"Expected TimeSeries or TimeSeriesTensor, got {type(ts)}")
