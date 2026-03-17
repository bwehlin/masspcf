#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import numpy.testing as npt
import pytest

import masspcf as mpcf
from masspcf.pcf import Pcf


class TestCopyConstructor:
    def test_copy_from_pcf_f32(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        g = Pcf(f)
        assert g.ttype == np.float32
        assert g.vtype == np.float32
        npt.assert_array_equal(f.to_numpy(), g.to_numpy())

    def test_copy_from_pcf_f64(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64))
        g = Pcf(f)
        assert g.ttype == np.float64
        assert g.vtype == np.float64
        npt.assert_array_equal(f.to_numpy(), g.to_numpy())


class TestIntArrayInput:
    def test_int64_array(self):
        arr = np.array([[0, 1], [2, 3]], dtype=np.int64)
        f = Pcf(arr)
        assert f.ttype == np.int64
        assert f.vtype == np.int64
        assert f.size() == 2

    def test_int32_array(self):
        arr = np.array([[0, 1], [2, 3]], dtype=np.int32)
        f = Pcf(arr)
        assert f.ttype == np.int32
        assert f.vtype == np.int32
        assert f.size() == 2

    def test_unsupported_array_dtype_raises(self):
        arr = np.array([[0, 1], [2, 3]], dtype=np.complex128)
        with pytest.raises(ValueError, match="Unsupported array type"):
            Pcf(arr)


class TestListInput:
    def test_list_default_dtype(self):
        f = Pcf([[0.0, 1.0], [2.0, 3.0]])
        assert f.ttype == np.float64
        assert f.vtype == np.float64
        assert f.size() == 2

    def test_list_f32(self):
        f = Pcf([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        assert f.ttype == np.float32
        assert f.vtype == np.float32
        npt.assert_array_almost_equal(
            f.to_numpy(), [[0.0, 1.0], [2.0, 3.0]]
        )

    def test_list_f64(self):
        f = Pcf([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        assert f.ttype == np.float64
        assert f.vtype == np.float64
        npt.assert_array_almost_equal(
            f.to_numpy(), [[0.0, 1.0], [2.0, 3.0]]
        )

    def test_list_int32(self):
        f = Pcf([[0, 1], [2, 3]], dtype=np.int32)
        assert f.ttype == np.int32
        assert f.vtype == np.int32
        npt.assert_array_equal(f.to_numpy(), [[0, 1], [2, 3]])

    def test_list_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            Pcf([[0.0, 1.0]], dtype=np.complex128)


class TestUnsupportedInputType:
    def test_string_raises(self):
        with pytest.raises(ValueError, match="unsupported input data"):
            Pcf("not a valid input")

    def test_dict_raises(self):
        with pytest.raises(ValueError, match="unsupported input data"):
            Pcf({"a": 1})


class TestDtypeConstructor:
    def test_ndarray_with_pcf32_dtype(self):
        arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        f = Pcf(arr, dtype=mpcf.pcf32)
        assert f.ttype == np.float32

    def test_ndarray_with_pcf64_dtype(self):
        arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        f = Pcf(arr, dtype=mpcf.pcf64)
        assert f.ttype == np.float64


class TestInternalMethods:
    def test_get_time_type_f32(self):
        f = Pcf(np.array([[0.0, 1.0]], dtype=np.float32))
        assert isinstance(f._get_time_type(), str)

    def test_get_value_type_f32(self):
        f = Pcf(np.array([[0.0, 1.0]], dtype=np.float32))
        assert isinstance(f._get_value_type(), str)

    def test_get_time_value_type_f32(self):
        f = Pcf(np.array([[0.0, 1.0]], dtype=np.float32))
        result = f._get_time_value_type()
        assert "_" in result


class TestTruediv:
    def test_div_scalar_f32(self):
        f = Pcf(np.array([[0.0, 4.0], [2.0, 8.0]], dtype=np.float32))
        f = f / 2.0
        result = f.to_numpy()
        npt.assert_array_almost_equal(result[:, 1], [2.0, 4.0])

    def test_div_scalar_f64(self):
        f = Pcf(np.array([[0.0, 4.0], [2.0, 8.0]], dtype=np.float64))
        f = f / 2.0
        result = f.to_numpy()
        npt.assert_array_almost_equal(result[:, 1], [2.0, 4.0])


class TestStr:
    def test_str_f32(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        s = str(f)
        assert "PCF" in s
        assert "float32" in s
        assert "size=2" in s

    def test_str_f64(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64))
        s = str(f)
        assert "float64" in s


class TestSize:
    def test_size(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
        assert f.size() == 3


class TestArrayProtocol:
    def test_array_with_dtype(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        arr = np.array(f, dtype=np.float64)
        assert arr.dtype == np.float64
        npt.assert_array_almost_equal(arr, [[0.0, 1.0], [2.0, 3.0]])

    def test_array_without_dtype(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        arr = np.array(f)
        assert arr.dtype == np.float32
        assert arr.shape == (2, 2)
        npt.assert_array_almost_equal(arr, [[0.0, 1.0], [2.0, 3.0]])


class TestEquality:
    def test_eq_same(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        g = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        assert f == g

    def test_eq_different(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        g = Pcf(np.array([[0.0, 1.0], [2.0, 9.0]], dtype=np.float32))
        assert not (f == g)

