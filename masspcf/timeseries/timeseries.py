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
from ..typing import (
    ts32,
    ts64,
)


def _datetime_unit(dt_or_arr):
    """Extract the unit string from a datetime64/timedelta64 scalar or array."""
    return np.datetime_data(np.dtype(dt_or_arr.dtype))[0]


class _DateTimeConverter:
    """Converts between numpy datetime64 and int64 ticks for the C++ chrono path.

    The C++ side stores start_time/time_step as float seconds. Query times
    are passed as int64 ticks + unit string so that C++ can construct the
    appropriate std::chrono::duration and perform exact integer subtraction
    before converting to float.
    """

    def __init__(self, start_time, time_step):
        self.start_time_dt = start_time
        self.time_step_td = time_step
        self._unit_str = _datetime_unit(time_step)
        self._unit_td = np.timedelta64(1, self._unit_str)

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
    """Convert a datetime64 scalar to (int64_ticks, unit_string)."""
    unit = _datetime_unit(t)
    ticks = int(t.astype(f"datetime64[{unit}]").view("int64"))
    return ticks, unit


def _dt64_array_to_ticks(arr):
    """Convert a datetime64 array to (int64_ticks_array, unit_string)."""
    unit = _datetime_unit(arr)
    ticks = arr.astype(f"datetime64[{unit}]").view("int64").astype("int64")
    return ticks, unit


_TS_CPP_TO_DTYPE = {
    cpp.TimeSeries32Tensor: ts32,
    cpp.TimeSeries64Tensor: ts64,
}


class TimeSeries:
    """A time series stored as a piecewise constant function with time metadata.

    Each ``TimeSeries`` wraps a :class:`~masspcf.Pcf` together with a
    ``start_time`` (the real-world time of the first sample) and a
    ``time_step`` (the duration between samples).

    Construction forms:

    ``TimeSeries(times, values)``
        Explicit time points. *times* is a 1-D array of sample times
        (float or ``datetime64``), *values* a 1-D array of sample values.
        ``start_time`` is inferred from ``times[0]``.

    ``TimeSeries(values, *, start_time=0.0, time_step=1.0)``
        Regular sampling. *values* is a 1-D array; sample *i* is placed
        at ``start_time + i * time_step``.

    ``TimeSeries(existing)``
        Copy an existing ``TimeSeries``.

    Parameters
    ----------
    start_time : float, int, or numpy.datetime64, optional
        Start time for regularly-sampled construction. Default 0.0.
    time_step : float, int, or numpy.timedelta64, optional
        Spacing between samples for regularly-sampled construction.
        Default 1.0. Not used when *times* is provided.
    dtype : masspcf.dtype, optional
        ``ts32`` or ``ts64``. If ``None``, inferred from data.
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
                 start_time=None, time_step=1.0, dtype=None):
        # --- Copy / C++ wrap ---
        if values is None and start_time is None:
            if isinstance(times_or_values, TimeSeries):
                self._data = times_or_values._data
                self.dtype = times_or_values.dtype
                self._start_time_raw = times_or_values._start_time_raw
                self._dt_converter = times_or_values._dt_converter
                return
            if isinstance(times_or_values, tuple(self._CPP_TO_DTYPE.keys())):
                self._data = times_or_values
                self.dtype = self._CPP_TO_DTYPE[type(times_or_values)]
                self._start_time_raw = times_or_values.start_time
                self._dt_converter = None
                return

        dt_converter = None

        # Resolve dtype for values
        def _resolve_dtype(arr):
            if dtype is not None:
                return self._DTYPE_TO_NP.get(dtype, np.float64)
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
            if len(times) != values.shape[0]:
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
                ticks = dt_converter.array_to_int64(times)
                self._data = cpp_cls(
                    ticks, values,
                    dt_converter.step_ticks(), dt_converter._unit_str)
                start_time_raw = times[0]
            else:
                # Store real offsets from start as PCF breakpoints,
                # time_step=1 so evaluate is just pcf_t = query - start
                times_f = times.astype(np_dtype)
                self._data = cpp_cls(times_f, values)
                start_time_raw = float(times_f[0])

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

            if isinstance(start_time, np.datetime64):
                if not isinstance(time_step, np.timedelta64):
                    raise TypeError(
                        "time_step must be timedelta64 when start_time "
                        "is datetime64")
                if time_step <= np.timedelta64(0):
                    raise ValueError("time_step must be positive")
                dt_converter = _DateTimeConverter(start_time, time_step)
                self._data = cpp_cls(
                    vals, dt_converter.start_ticks(),
                    dt_converter.step_ticks(), dt_converter._unit_str)
                start_time_raw = start_time
            else:
                # Build explicit times: start + i * step
                start_f = float(start_time)
                step_f = float(time_step)
                if step_f <= 0:
                    raise ValueError("time_step must be positive")
                n = len(vals)
                times_arr = (start_f + np.arange(n, dtype=np_dtype) * step_f)
                self._data = cpp_cls(times_arr, vals)
                start_time_raw = start_f

        self.dtype = self._CPP_TO_DTYPE.get(type(self._data), ts64)
        self._start_time_raw = start_time_raw
        self._dt_converter = dt_converter

    def __call__(self, t):
        """Evaluate the time series at the given time(s).

        Parameters
        ----------
        t : float, int, numpy.datetime64, or numpy.ndarray
            Query time(s). If ``start_time`` is a datetime64, *t* should
            also be datetime64 (converted to PCF-internal time via the
            datetime converter for full precision).

        Returns
        -------
        float or numpy.ndarray
            Value(s) at the queried time(s). NaN for times outside the
            series domain.
        """
        if isinstance(t, np.datetime64):
            ticks, unit = _dt64_to_ticks(t)
            return self._data(ticks, unit)
        if isinstance(t, np.ndarray) and np.issubdtype(t.dtype, np.datetime64):
            ticks, unit = _dt64_array_to_ticks(t)
            return self._data(ticks, unit)
        if isinstance(t, np.ndarray):
            return self._data(t)
        if isinstance(t, (int, float)):
            return self._data(t)
        if isinstance(t, list):
            return self._data(np.asarray(t, dtype=np.float64))
        raise TypeError(f"Cannot evaluate TimeSeries at type {type(t)}")

    @property
    def start_time(self):
        """The real-world time of the first sample."""
        return self._start_time_raw

    @property
    def end_time(self):
        """The real-world time of the last sample."""
        return self._data.end_time

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
        # TODO: expose values directly from C++ without copy
        nc = self._data.n_channels
        nt = self._data.n_times
        if nc == 1:
            return np.array([self._data(float(self._data.start_time
                             + self._data.time_step * i))
                             for i in range(nt)])
        # For now, evaluate at each breakpoint time
        result = np.empty((nt, nc))
        for i in range(nt):
            t = self._data.start_time + self._data.time_step * i
            vals = self._data(float(t))
            if isinstance(vals, np.ndarray):
                result[i] = vals
            else:
                result[i, 0] = vals
        return result

    @property
    def times(self):
        """The real-world times for each sample."""
        dtc = self._dt_converter
        if dtc is not None:
            steps = np.arange(self._data.n_times)
            return dtc.start_time_dt + steps * dtc.time_step_td
        return self._data.start_time + self._data._internal_times

    def __repr__(self):
        nc = self.n_channels
        chan_str = f", n_channels={nc}" if nc > 1 else ""
        return (
            f"TimeSeries(start_time={self.start_time}, "
            f"n_times={self.n_times}{chan_str}, dtype={self.dtype})"
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
        self.dtype = _TS_CPP_TO_DTYPE[type(self._data)]

    def __call__(self, t):
        """Evaluate every series at the given time(s).

        Supports ``datetime64`` scalars and arrays. Datetime values are
        passed as int64 ticks + unit string to C++, where each element's
        chrono evaluate converts to seconds independently.
        """
        if isinstance(t, np.datetime64):
            ticks, unit = _dt64_to_ticks(t)
            return np.asarray(self._data(ticks, unit))
        if isinstance(t, np.ndarray) and np.issubdtype(t.dtype, np.datetime64):
            ticks, unit = _dt64_array_to_ticks(t)
            return np.asarray(self._data(ticks, unit))
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
