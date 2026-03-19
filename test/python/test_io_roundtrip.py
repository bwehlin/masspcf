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

import io

import numpy as np

import masspcf as mpcf


def _roundtrip(tensor):
    buf = io.BytesIO()
    mpcf.save(tensor, buf)
    buf.seek(0)
    return mpcf.load(buf)


# --- Float tensors ---


def test_float32_tensor_roundtrip():
    original = mpcf.Float32Tensor(np.array([[1.0, 2.5], [3.0, -4.5]], dtype=np.float32))
    restored = _roundtrip(original)
    assert original.array_equal(restored)


def test_float64_tensor_roundtrip():
    original = mpcf.Float64Tensor(np.array([[1.0, 2.5], [3.0, -4.5]], dtype=np.float64))
    restored = _roundtrip(original)
    assert original.array_equal(restored)


def test_float32_tensor_1d():
    original = mpcf.Float32Tensor(np.array([1.5, 2.5, 3.5], dtype=np.float32))
    restored = _roundtrip(original)
    assert original.array_equal(restored)


def test_float64_tensor_3d():
    original = mpcf.Float64Tensor(np.random.randn(2, 3, 4).astype(np.float64))
    restored = _roundtrip(original)
    assert original.array_equal(restored)


def test_float32_tensor_scalar():
    original = mpcf.Float32Tensor(np.array([42.0], dtype=np.float32))
    restored = _roundtrip(original)
    assert original.array_equal(restored)


# --- PCF tensors ---


def test_pcf32_tensor_roundtrip():
    original = mpcf.random.noisy_sin((3, 4), dtype=mpcf.pcf32)
    restored = _roundtrip(original)
    assert original.array_equal(restored)


def test_pcf64_tensor_roundtrip():
    original = mpcf.random.noisy_sin((3, 4), dtype=mpcf.pcf64)
    restored = _roundtrip(original)
    assert original.array_equal(restored)


def test_pcf32_tensor_1d():
    original = mpcf.random.noisy_cos((5,), dtype=mpcf.pcf32)
    restored = _roundtrip(original)
    assert original.array_equal(restored)


def test_pcf32_tensor_zeros():
    original = mpcf.zeros((2, 3), dtype=mpcf.pcf32)
    restored = _roundtrip(original)
    assert original.array_equal(restored)


def test_pcf64_tensor_zeros():
    original = mpcf.zeros((2, 3), dtype=mpcf.pcf64)
    restored = _roundtrip(original)
    assert original.array_equal(restored)


# --- PointCloud tensors ---


def test_point_cloud32_tensor_roundtrip():
    original = mpcf.zeros((3,), dtype=mpcf.pcloud32)
    original[0] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    original[1] = np.array([[5.0, 6.0]], dtype=np.float32)
    restored = _roundtrip(original)
    assert original.array_equal(restored)


def test_point_cloud64_tensor_roundtrip():
    original = mpcf.zeros((3,), dtype=mpcf.pcloud64)
    original[0] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    original[1] = np.array([[5.0, 6.0]], dtype=np.float64)
    restored = _roundtrip(original)
    assert original.array_equal(restored)


def test_point_cloud32_tensor_empty():
    original = mpcf.zeros((2,), dtype=mpcf.pcloud32)
    restored = _roundtrip(original)
    assert original.array_equal(restored)


# --- Barcode tensors ---


def test_barcode32_tensor_roundtrip():
    original = mpcf.zeros((2,), dtype=mpcf.barcode32)
    original[0] = np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float32)
    restored = _roundtrip(original)
    assert original.array_equal(restored)


def test_barcode64_tensor_roundtrip():
    original = mpcf.zeros((2,), dtype=mpcf.barcode64)
    original[0] = np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float64)
    restored = _roundtrip(original)
    assert original.array_equal(restored)


def test_barcode32_tensor_empty():
    original = mpcf.zeros((3,), dtype=mpcf.barcode32)
    restored = _roundtrip(original)
    assert original.array_equal(restored)


# --- Dtype preservation ---


def test_load_preserves_float32_type():
    original = mpcf.Float32Tensor(np.ones((2, 2), dtype=np.float32))
    restored = _roundtrip(original)
    assert type(restored) is mpcf.Float32Tensor
    assert restored.dtype == mpcf.f32


def test_load_preserves_float64_type():
    original = mpcf.Float64Tensor(np.ones((2, 2), dtype=np.float64))
    restored = _roundtrip(original)
    assert type(restored) is mpcf.Float64Tensor
    assert restored.dtype == mpcf.f64


def test_load_preserves_pcf32_type():
    original = mpcf.random.noisy_sin((2,), dtype=mpcf.pcf32)
    restored = _roundtrip(original)
    assert type(restored) is mpcf.Pcf32Tensor
    assert restored.dtype == mpcf.pcf32


def test_load_preserves_pcf64_type():
    original = mpcf.random.noisy_sin((2,), dtype=mpcf.pcf64)
    restored = _roundtrip(original)
    assert type(restored) is mpcf.Pcf64Tensor
    assert restored.dtype == mpcf.pcf64


# --- Shape preservation ---


def test_shape_preserved_2d():
    original = mpcf.Float32Tensor(np.arange(35, dtype=np.float32).reshape(5, 7))
    restored = _roundtrip(original)
    assert restored.shape == original.shape


def test_shape_preserved_3d():
    original = mpcf.Float64Tensor(np.arange(24, dtype=np.float64).reshape(2, 3, 4))
    restored = _roundtrip(original)
    assert restored.shape == original.shape


def test_shape_preserved_pcf():
    original = mpcf.random.noisy_sin((4, 5), dtype=mpcf.pcf32)
    restored = _roundtrip(original)
    assert restored.shape == original.shape
