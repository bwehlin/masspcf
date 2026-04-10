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

from . import _mpcf_cpp as cpp
from ._tensor_base import FunctionTensorMixin, Tensor, _tensor_from_nested
from .functional.pcf import Pcf
from .typing import (
    _MPCF_TO_NP,
    _NP_TO_MPCF,
    float32,
    float64,
    ts32,
    ts64,
)


def _timedelta64_to_float(val):
    """Convert numpy.timedelta64 to float seconds."""
    return val / np.timedelta64(1, "s")


def _infer_datetime_unit(time_step):
    """Infer the base unit for datetime conversions from a timedelta64.

    Returns a numpy.timedelta64 representing one unit, used to convert
    datetime64 values to integer counts of that unit. This avoids
    precision loss from large float epoch values.
    """
    # Use the resolution of the timedelta itself as our unit
    # e.g. timedelta64(10, 'ms') -> unit is timedelta64(1, 'ms')
    dtype_str = str(time_step.dtype)  # e.g. "timedelta64[ms]"
    unit = dtype_str.split("[")[1].rstrip("]")
    return np.timedelta64(1, unit)


class _DateTimeConverter:
    """Handles conversion between datetime64 domain and float domain.

    Uses the time_step's native unit (e.g. milliseconds for
    ``timedelta64(10, 'ms')``) as the float coordinate system.
    This preserves full precision because values like
    ``1704067200000.0`` (ms since Unix epoch for 2024) fit exactly
    in float64 (well within the 2^53 integer range).
    """

    def __init__(self, start_time, time_step):
        self.start_time_dt = start_time
        self.time_step_td = time_step
        self._unit = _infer_datetime_unit(time_step)
        unit_str = str(self._unit.dtype).split("[")[1].rstrip("]")
        self._zero = np.datetime64(0, unit_str)

    def start_time_float(self):
        """Return start_time as a float in the native unit."""
        return float((self.start_time_dt - self._zero) / self._unit)

    def time_step_float(self):
        """Return time_step as a float in the native unit."""
        return float(self.time_step_td / self._unit)

    def convert_scalar(self, t):
        """Convert a datetime64 scalar to a float in the native unit."""
        return float((t - self._zero) / self._unit)

    def convert_array(self, t):
        """Convert a datetime64 array to floats in the native unit."""
        return ((t - self._zero) / self._unit).astype(np.float64)


def _convert_start_time(start_time):
    """Convert start_time to float, handling datetime64."""
    if isinstance(start_time, np.datetime64):
        raise ValueError("datetime64 start_time requires a _DateTimeConverter")
    return float(start_time)


def _convert_time_step(time_step):
    """Convert time_step to float, handling timedelta64."""
    if isinstance(time_step, np.timedelta64):
        raise ValueError("timedelta64 time_step requires a _DateTimeConverter")
    return float(time_step)


_TS_CPP_TO_DTYPE = {
    cpp.TimeSeries32Tensor: ts32,
    cpp.TimeSeries64Tensor: ts64,
}


class TimeSeries:
    """A time series stored as a piecewise constant function with time metadata.

    Each ``TimeSeries`` wraps a :class:`~masspcf.Pcf` together with a
    ``start_time`` (the real-world time corresponding to PCF t=0) and a
    ``time_step`` (the real-world duration per PCF time unit).

    Parameters
    ----------
    data : numpy.ndarray, Pcf, or TimeSeries
        Input data.

        * **1-D array of values**: breakpoints placed at PCF times 0, 1, 2, ...
          Each value lasts for one ``time_step``.
        * **(n, 2) array of (time, value) pairs**: ``start_time`` is inferred
          from the first time value. Times are converted to PCF-internal
          coordinates using ``time_step``.
        * **Pcf**: wrap an existing PCF directly (must also supply
          start_time/time_step).
        * **TimeSeries**: copy.
    start_time : float, int, or numpy.datetime64, optional
        The real-world time corresponding to PCF t=0. For (n, 2) arrays
        this is overridden by the first time value. Default 0.0.
    time_step : float, int, or numpy.timedelta64, optional
        The real-world duration per PCF time unit. Default 1.0.
    dtype : masspcf.dtype, optional
        ``ts32`` or ``ts64``. If ``None``, inferred from input data.
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

    def __init__(self, data, *, start_time=0.0, time_step=1.0, dtype=None):
        if isinstance(data, TimeSeries):
            self._data = data._data
            self.dtype = data.dtype
            self._start_time_raw = data._start_time_raw
            self._time_step_raw = data._time_step_raw
            self._dt_converter = data._dt_converter
            return

        start_time_raw = start_time
        time_step_raw = time_step
        dt_converter = None

        if isinstance(start_time, np.datetime64) and isinstance(time_step, np.timedelta64):
            dt_converter = _DateTimeConverter(start_time, time_step)
            start_time_f = dt_converter.start_time_float()
            time_step_f = dt_converter.time_step_float()
        else:
            start_time_f = _convert_start_time(start_time)
            time_step_f = _convert_time_step(time_step)

        if isinstance(data, Pcf):
            np_dtype = self._DTYPE_TO_NP.get(dtype) if dtype else None
            if np_dtype is None:
                np_dtype = _MPCF_TO_NP.get(data.ttype, np.float64)
            cpp_cls = self._NP_TO_CPP_TS[np_dtype]
            self._data = cpp_cls(data._data, np_dtype(start_time_f), np_dtype(time_step_f))

        elif isinstance(data, tuple(self._CPP_TO_DTYPE.keys())):
            self._data = data
            start_time_raw = data.start_time
            time_step_raw = data.time_step

        elif isinstance(data, np.ndarray):
            if dtype is not None:
                np_dtype = self._DTYPE_TO_NP.get(dtype)
                if np_dtype and data.dtype != np_dtype:
                    data = data.astype(np_dtype)
            np_dtype = data.dtype.type
            if np_dtype not in self._NP_TO_CPP_TS:
                data = data.astype(np.float64)
                np_dtype = np.float64
            cpp_cls = self._NP_TO_CPP_TS[np_dtype]
            self._data = cpp_cls(data, np_dtype(start_time_f), np_dtype(time_step_f))
            # For (n,2) input, start_time is inferred by C++ from the first time
            if dt_converter is None:
                start_time_raw = self._data.start_time
                time_step_raw = self._data.time_step

        elif isinstance(data, list):
            arr = np.array(data, dtype=np.float64 if dtype is None else self._DTYPE_TO_NP.get(dtype, np.float64))
            np_dtype = arr.dtype.type
            cpp_cls = self._NP_TO_CPP_TS.get(np_dtype, cpp.TimeSeries_f64_f64)
            self._data = cpp_cls(arr, np_dtype(start_time_f), np_dtype(time_step_f))

        else:
            raise TypeError(f"Cannot create TimeSeries from {type(data)}")

        self.dtype = self._CPP_TO_DTYPE.get(type(self._data), ts64)
        self._start_time_raw = start_time_raw
        self._time_step_raw = time_step_raw
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
        dtc = self._dt_converter
        if isinstance(t, np.datetime64):
            if dtc is not None:
                return self._data(dtc.convert_scalar(t))
            return self._data(float(t))
        if isinstance(t, np.ndarray):
            if np.issubdtype(t.dtype, np.datetime64):
                if dtc is not None:
                    t = dtc.convert_array(t)
                else:
                    t = t.astype(np.float64)
            return self._data(t)
        if isinstance(t, (int, float)):
            return self._data(t)
        if isinstance(t, list):
            return self._data(np.asarray(t, dtype=np.float64))
        raise TypeError(f"Cannot evaluate TimeSeries at type {type(t)}")

    @property
    def pcf(self):
        """The underlying :class:`~masspcf.Pcf`."""
        return Pcf(self._data.pcf)

    @property
    def start_time(self):
        """The real-world time corresponding to PCF t=0."""
        return self._start_time_raw

    @property
    def time_step(self):
        """The real-world duration per PCF time unit."""
        return self._time_step_raw

    @property
    def end_time(self):
        """End time of the series (start_time + last_breakpoint * time_step)."""
        dtc = self._dt_converter
        if dtc is not None:
            # C++ end_time is in the converter's native unit (e.g. ms)
            return dtc.start_time_dt + int(round(
                self._data.end_time - self._data.start_time
            )) * dtc._unit
        return self._data.end_time

    @property
    def size(self):
        """Number of breakpoints in the underlying PCF."""
        return self._data.size

    @property
    def values(self):
        """The PCF values as a 1-D numpy array."""
        arr = np.asarray(self._data.pcf)
        return arr[:, 1]

    @property
    def times(self):
        """Reconstructed real-world times for each breakpoint."""
        arr = np.asarray(self._data.pcf)
        pcf_times = arr[:, 0]
        dtc = self._dt_converter
        if dtc is not None:
            # PCF times * time_step_float gives offsets in the native unit
            offsets = np.round(pcf_times * dtc.time_step_float()).astype("int64")
            return dtc.start_time_dt + offsets * dtc._unit
        return self._data.start_time + pcf_times * self._data.time_step

    def __repr__(self):
        return (
            f"TimeSeries(start_time={self.start_time}, time_step={self.time_step}, "
            f"size={self.size}, dtype={self.dtype})"
        )

    def __len__(self):
        return self.size

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
        dt_converter = None
        if isinstance(data, TimeSeriesTensor):
            dt_converter = data._dt_converter
            data = data._data
        elif isinstance(data, (list, tuple)):
            from ._tensor_base import _infer_shape_and_flatten
            _, flat_ts = _infer_shape_and_flatten(data)
            if flat_ts:
                for ts in flat_ts:
                    if getattr(ts, "_dt_converter", None) is not None:
                        dt_converter = ts._dt_converter
                        break
            data = _tensor_from_nested(data, {
                cpp.TimeSeries_f32_f32: cpp.TimeSeries32Tensor,
                cpp.TimeSeries_f64_f64: cpp.TimeSeries64Tensor,
            })
        elif not isinstance(data, (cpp.TimeSeries32Tensor, cpp.TimeSeries64Tensor)):
            raise TypeError(f"Cannot create TimeSeriesTensor from {type(data)}")
        self._data = data
        self.dtype = _TS_CPP_TO_DTYPE[type(self._data)]
        self._dt_converter = dt_converter

    def __call__(self, t):
        """Evaluate every series at the given time(s).

        Supports ``datetime64`` scalars and arrays in addition to the
        numeric types handled by the base class. Datetime values are
        converted to floats in the time_step's native unit and passed
        directly to C++ tensor evaluation -- no Python loop.
        """
        dtc = self._dt_converter
        if isinstance(t, np.datetime64):
            if dtc is not None:
                t = dtc.convert_scalar(t)
            else:
                t = float(t)
        elif isinstance(t, np.ndarray) and np.issubdtype(t.dtype, np.datetime64):
            if dtc is not None:
                t = dtc.convert_array(t)
            else:
                t = t.astype(np.float64)
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
