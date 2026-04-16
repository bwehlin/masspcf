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
from masspcf.persistence import Barcode, compute_persistent_homology
from masspcf.persistence.ph_tensor import BarcodeTensor
from masspcf.tensor import FloatTensor, PcfTensor
from masspcf.timeseries import (
    CallableInterpolation,
    TensorTimeSeries,
    TimeSeries,
)
from masspcf.typing import (
    barcode32,
    barcode64,
    float32,
    float64,
    pcf32,
    pcf64,
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
        assert ts.dtype is pcf32
        assert ts.n_times == 2
        assert ts.n_channels == 1

    def test_dtype_f64(self):
        p1 = _pcf([[0.0, 1.0], [1.0, 0.0]], dt=np.float64)
        p2 = _pcf([[0.0, 2.0], [1.0, 0.0]], dt=np.float64)
        ts = TimeSeries([0.0, 1.0], [p1, p2])
        assert ts.dtype is pcf64

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
        assert ts.dtype is barcode32
        assert ts.n_times == 2

    def test_dtype_f64(self):
        b1 = _bc([[0.0, 1.0]], dt=np.float64)
        b2 = _bc([[0.5, 2.0]], dt=np.float64)
        ts = TimeSeries([0.0, 1.0], [b1, b2])
        assert ts.dtype is barcode64

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


class TestPointCloudValuedTimeSeries:
    @staticmethod
    def _random_clouds(n, m=12, d=2, dtype=np.float32, seed=0):
        rng = np.random.default_rng(seed)
        return [FloatTensor(rng.standard_normal((m, d)).astype(dtype))
                for _ in range(n)]

    def test_construct_from_float_tensor_list(self):
        clouds = self._random_clouds(4)
        ts = TensorTimeSeries([0.0, 1.0, 2.0, 3.0], clouds)
        assert ts.dtype is float32
        assert ts.n_times == 4
        assert ts.n_channels == 1

    def test_construct_from_ndarrays(self):
        rng = np.random.default_rng(1)
        clouds_np = [rng.standard_normal((8, 2)).astype(np.float32)
                     for _ in range(3)]
        ts = TensorTimeSeries([0.0, 1.0, 2.0], clouds_np)
        assert ts.dtype is float32

    def test_construct_f64(self):
        clouds = self._random_clouds(3, dtype=np.float64)
        ts = TensorTimeSeries([0.0, 1.0, 2.0], clouds)
        assert ts.dtype is float64

    def test_evaluate_returns_float_tensor(self):
        clouds = self._random_clouds(3)
        ts = TensorTimeSeries([0.0, 1.0, 2.0], clouds)
        out = ts(0.5)
        assert isinstance(out, FloatTensor)
        # nearest picks left breakpoint: cloud at t=0
        np.testing.assert_array_equal(np.asarray(out), np.asarray(clouds[0]))

    def test_linear_interpolation_blocked(self):
        clouds = self._random_clouds(2)
        ts = TensorTimeSeries([0.0, 1.0], clouds)
        with pytest.raises(Exception, match="linear interpolation"):
            ts.interpolation = 'linear'


class TestPcfTensorValuedTimeSeries:
    def test_construct_from_pcf_tensor_list(self):
        p1 = Pcf(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
        p2 = Pcf(np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32))
        t1 = PcfTensor([p1, p2])
        t2 = PcfTensor([p2, p1])
        ts = TensorTimeSeries([0.0, 1.0], [t1, t2])
        assert ts.dtype is pcf32
        assert ts.n_times == 2

    def test_evaluate_returns_pcf_tensor(self):
        p1 = Pcf(np.array([[0.0, 1.0]], dtype=np.float32))
        p2 = Pcf(np.array([[0.0, 2.0]], dtype=np.float32))
        t1 = PcfTensor([p1, p2])
        t2 = PcfTensor([p2, p1])
        ts = TensorTimeSeries([0.0, 1.0], [t1, t2])
        out = ts(0.3)
        assert isinstance(out, PcfTensor)


class TestPersistentHomologyOnTimeSeries:
    @staticmethod
    def _random_clouds(n, m=12, d=2, dtype=np.float32, seed=0):
        rng = np.random.default_rng(seed)
        return [FloatTensor(rng.standard_normal((m, d)).astype(dtype))
                for _ in range(n)]

    def test_output_dtype_and_shape(self):
        clouds = self._random_clouds(5)
        pc_ts = TensorTimeSeries([0.0, 1.0, 2.0, 3.0, 4.0], clouds)
        bc_ts = compute_persistent_homology(pc_ts, max_dim=1)
        assert bc_ts.dtype is barcode32
        assert bc_ts.n_times == 5
        assert bc_ts.n_channels == 1

    def test_per_timestep_matches_per_cloud(self):
        clouds = self._random_clouds(5, seed=42)
        pc_ts = TensorTimeSeries(np.arange(5.0), clouds)
        bc_ts = compute_persistent_homology(pc_ts, max_dim=1)

        for i, cloud in enumerate(clouds):
            at_ti = bc_ts(float(i))
            assert isinstance(at_ti, BarcodeTensor)
            assert at_ti.shape == (2,)
            direct = compute_persistent_homology(cloud, max_dim=1)
            assert at_ti[0].is_isomorphic_to(direct[0])
            assert at_ti[1].is_isomorphic_to(direct[1])

    def test_max_dim_controls_n_h(self):
        clouds = self._random_clouds(3)
        pc_ts = TensorTimeSeries([0.0, 1.0, 2.0], clouds)
        bc_ts0 = compute_persistent_homology(pc_ts, max_dim=0)
        assert bc_ts0(1.0).shape == (1,)
        bc_ts2 = compute_persistent_homology(pc_ts, max_dim=2)
        assert bc_ts2(1.0).shape == (3,)

    def test_f64_input(self):
        clouds = self._random_clouds(3, dtype=np.float64)
        pc_ts = TensorTimeSeries([0.0, 1.0, 2.0], clouds)
        bc_ts = compute_persistent_homology(pc_ts, max_dim=1)
        assert bc_ts.dtype is barcode64

    def test_preserves_time_axis(self):
        clouds = self._random_clouds(4)
        times = [0.5, 1.5, 3.0, 4.0]
        pc_ts = TensorTimeSeries(times, clouds)
        bc_ts = compute_persistent_homology(pc_ts, max_dim=1)
        np.testing.assert_allclose(bc_ts.times, times)


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
        assert ts.dtype is float64
