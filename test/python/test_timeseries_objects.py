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

"""Tests for object-valued TimeSeries (Pcf, Barcode) and pluggable
interpolation strategies including Python-callable strategies."""

import numpy as np
import pytest

from masspcf.functional import Pcf
from masspcf.persistence import Barcode
from masspcf.timeseries import (
    CallableInterpolation,
    TimeSeries,
)
from masspcf.typing import (
    ts64,
    ts_barcode32,
    ts_barcode64,
    ts_pcf32,
    ts_pcf64,
)


def _pcf(points, dt=np.float32):
    return Pcf(np.asarray(points, dtype=dt))


def _bc(pairs, dt=np.float32):
    return Barcode(np.asarray(pairs, dtype=dt))


class TestPcfValuedTimeSeries:
    def test_construct_from_pcf_list(self):
        p1 = _pcf([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
        p2 = _pcf([[0.0, 3.0], [1.0, 1.0], [2.0, 0.0]])
        ts = TimeSeries([0.0, 1.0], [p1, p2])
        assert ts.dtype is ts_pcf32
        assert ts.n_times == 2
        assert ts.n_channels == 1

    def test_dtype_f64(self):
        p1 = _pcf([[0.0, 1.0], [1.0, 0.0]], dt=np.float64)
        p2 = _pcf([[0.0, 2.0], [1.0, 0.0]], dt=np.float64)
        ts = TimeSeries([0.0, 1.0], [p1, p2])
        assert ts.dtype is ts_pcf64

    def test_evaluate_nearest_returns_left_breakpoint(self):
        p1 = _pcf([[0.0, 1.0], [1.0, 0.0]])
        p2 = _pcf([[0.0, 5.0], [1.0, 0.0]])
        ts = TimeSeries([0.0, 1.0], [p1, p2])
        out = ts(0.3)
        assert isinstance(out, Pcf)
        # Nearest picks left breakpoint (p1): value at x=0 is 1.0
        assert out(0.5) == pytest.approx(1.0)

    def test_evaluate_linear_blends_pcfs(self):
        # Linear blend of two PCFs at alpha=0.5 should produce a Pcf whose
        # value at any point is the average of the two inputs at that point.
        p1 = _pcf([[0.0, 1.0], [1.0, 0.0]])
        p2 = _pcf([[0.0, 5.0], [1.0, 0.0]])
        ts = TimeSeries([0.0, 1.0], [p1, p2], interpolation='linear')
        blended = ts(0.5)
        assert isinstance(blended, Pcf)
        # At x=0.5 (inside first breakpoint interval), p1 = 1.0, p2 = 5.0
        # Linear blend at alpha=0.5: 0.5*1 + 0.5*5 = 3.0
        assert blended(0.5) == pytest.approx(3.0)

    def test_values_returns_python_wrappers(self):
        p1 = _pcf([[0.0, 1.0], [1.0, 0.0]])
        p2 = _pcf([[0.0, 2.0], [1.0, 0.0]])
        ts = TimeSeries([0.0, 1.0], [p1, p2])
        vals = ts.values
        assert isinstance(vals, list)
        assert len(vals) == 2
        assert all(isinstance(v, Pcf) for v in vals)


class TestBarcodeValuedTimeSeries:
    def test_construct_from_barcode_list(self):
        b1 = _bc([[0.0, 1.0], [0.5, 2.0]])
        b2 = _bc([[0.0, 2.0], [0.3, 3.0]])
        ts = TimeSeries([0.0, 1.0], [b1, b2])
        assert ts.dtype is ts_barcode32
        assert ts.n_times == 2

    def test_dtype_f64(self):
        b1 = _bc([[0.0, 1.0]], dt=np.float64)
        b2 = _bc([[0.5, 2.0]], dt=np.float64)
        ts = TimeSeries([0.0, 1.0], [b1, b2])
        assert ts.dtype is ts_barcode64

    def test_evaluate_nearest_returns_left_breakpoint(self):
        b1 = _bc([[0.0, 1.0], [0.5, 2.0]])
        b2 = _bc([[0.0, 2.0], [0.3, 3.0]])
        ts = TimeSeries([0.0, 1.0], [b1, b2])
        out = ts(0.3)
        assert isinstance(out, Barcode)
        assert out.is_isomorphic_to(b1)

    def test_linear_interpolation_not_available(self):
        b1 = _bc([[0.0, 1.0]])
        b2 = _bc([[0.5, 2.0]])
        with pytest.raises(Exception, match="linear interpolation"):
            TimeSeries([0.0, 1.0], [b1, b2], interpolation='linear')


class TestCallableInterpolation:
    def test_scalar_step_from_left(self):
        def step_from_left(queries, t_lefts, t_rights, v_lefts, v_rights):
            return list(v_lefts)

        ts = TimeSeries([0.0, 1.0, 2.0], [1.0, 2.0, 3.0],
                        interpolation=CallableInterpolation(step_from_left))
        assert ts(0.3) == pytest.approx(1.0)
        assert ts(0.9) == pytest.approx(1.0)
        assert ts(1.7) == pytest.approx(2.0)

    def test_scalar_step_from_right(self):
        def step_from_right(queries, t_lefts, t_rights, v_lefts, v_rights):
            return list(v_rights)

        ts = TimeSeries([0.0, 1.0, 2.0], [1.0, 2.0, 3.0],
                        interpolation=CallableInterpolation(step_from_right))
        assert ts(0.3) == pytest.approx(2.0)
        assert ts(1.1) == pytest.approx(3.0)

    def test_custom_callable_receives_batched_inputs(self):
        received_lengths = []

        def tracker(queries, t_lefts, t_rights, v_lefts, v_rights):
            received_lengths.append(len(queries))
            assert len(t_lefts) == len(queries)
            assert len(v_lefts) == len(queries)
            return list(v_lefts)

        ts = TimeSeries([0.0, 1.0, 2.0], [1.0, 2.0, 3.0],
                        interpolation=CallableInterpolation(tracker))
        ts(np.array([0.1, 0.5, 1.3, 1.8], dtype=np.float64))
        assert sum(received_lengths) == 4

    def test_custom_callable_requires_callable(self):
        with pytest.raises(TypeError):
            CallableInterpolation("not a function")


class TestInterpolationReporting:
    def test_string_interpolation_reports_string(self):
        ts = TimeSeries([0.0, 1.0], [1.0, 2.0], interpolation='linear')
        _ = ts(0.5)  # force lazy strategy materialization
        assert ts.interpolation == 'linear'

    def test_custom_strategy_reports_custom(self):
        def identity(queries, tl, tr, vl, vr):
            return list(vl)
        ts = TimeSeries([0.0, 1.0], [1.0, 2.0],
                        interpolation=CallableInterpolation(identity))
        assert ts.interpolation == 'custom'

    def test_setter_accepts_strategy(self):
        def identity(queries, tl, tr, vl, vr):
            return list(vl)
        ts = TimeSeries([0.0, 1.0], [1.0, 2.0])
        ts.interpolation = CallableInterpolation(identity)
        assert ts.interpolation == 'custom'

    def test_setter_accepts_string(self):
        def identity(queries, tl, tr, vl, vr):
            return list(vl)
        ts = TimeSeries([0.0, 1.0], [1.0, 2.0],
                        interpolation=CallableInterpolation(identity))
        ts.interpolation = 'nearest'
        _ = ts(0.5)
        assert ts.interpolation == 'nearest'


class TestBackwardCompat:
    def test_scalar_string_nearest_unchanged(self):
        ts = TimeSeries([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
        assert ts(0.5) == pytest.approx(1.0)
        assert ts(1.5) == pytest.approx(2.0)

    def test_scalar_string_linear_unchanged(self):
        ts = TimeSeries([0.0, 1.0], [1.0, 2.0], interpolation='linear')
        assert ts(0.5) == pytest.approx(1.5)

    def test_scalar_dtype_detection_unchanged(self):
        ts = TimeSeries([0.0, 1.0], [1.0, 2.0])
        assert ts.dtype is ts64
