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
import masspcf as mpcf


def test_noisy_trig():
    X = mpcf.random.noisy_sin((10, 20))
    assert X.shape == (10, 20)


def test_global_seed_determinism():
    mpcf.random.seed(42)
    A = mpcf.random.noisy_sin((5, 10))

    mpcf.random.seed(42)
    B = mpcf.random.noisy_sin((5, 10))

    assert A.array_equal(B)


def test_global_seed_different_seeds_differ():
    mpcf.random.seed(42)
    A = mpcf.random.noisy_sin((5, 10))

    mpcf.random.seed(99)
    B = mpcf.random.noisy_sin((5, 10))

    assert not A.array_equal(B)


def test_generator_determinism():
    gen = mpcf.random.Generator(seed=123)
    A = mpcf.random.noisy_sin((5, 10), generator=gen)

    gen2 = mpcf.random.Generator(seed=123)
    B = mpcf.random.noisy_sin((5, 10), generator=gen2)

    assert A.array_equal(B)


def test_generator_noisy_cos():
    gen = mpcf.random.Generator(seed=77)
    A = mpcf.random.noisy_cos((3, 4), generator=gen)

    gen2 = mpcf.random.Generator(seed=77)
    B = mpcf.random.noisy_cos((3, 4), generator=gen2)

    assert A.array_equal(B)


def test_determinism_pcf64():
    gen = mpcf.random.Generator(seed=42)
    A = mpcf.random.noisy_sin((3, 4), dtype=mpcf.pcf64, generator=gen)

    gen2 = mpcf.random.Generator(seed=42)
    B = mpcf.random.noisy_sin((3, 4), dtype=mpcf.pcf64, generator=gen2)

    assert A.array_equal(B)


# --- random_pcf tests ---


def test_random_pcf_shape():
    gen = mpcf.random.Generator(seed=1)
    A = mpcf.random.random_pcf((5, 3), generator=gen)
    assert A.shape == (5, 3)


def test_random_pcf_breakpoint_count():
    gen = mpcf.random.Generator(seed=2)
    A = mpcf.random.random_pcf((20,), n_range=(15, 50), generator=gen)
    for i in range(20):
        n = len(A[i].to_numpy())
        assert 15 <= n <= 50, f"PCF {i} has {n} breakpoints"


def test_random_pcf_fixed_n():
    gen = mpcf.random.Generator(seed=3)
    A = mpcf.random.random_pcf((10,), n_range=42, generator=gen)
    for i in range(10):
        assert len(A[i].to_numpy()) == 42


def test_random_pcf_final_value_zero():
    gen = mpcf.random.Generator(seed=4)
    A = mpcf.random.random_pcf((30,), generator=gen)
    for i in range(30):
        arr = A[i].to_numpy()
        assert arr[-1, 1] == 0.0, f"PCF {i} final value = {arr[-1, 1]}"


def test_random_pcf_times_start_at_zero():
    gen = mpcf.random.Generator(seed=5)
    A = mpcf.random.random_pcf((20,), generator=gen)
    for i in range(20):
        arr = A[i].to_numpy()
        assert arr[0, 0] == 0.0


def test_random_pcf_times_increasing():
    gen = mpcf.random.Generator(seed=6)
    A = mpcf.random.random_pcf((20,), generator=gen)
    for i in range(20):
        times = A[i].to_numpy()[:, 0]
        assert np.all(np.diff(times) >= 0), f"PCF {i} times not increasing"


def test_random_pcf_determinism():
    gen1 = mpcf.random.Generator(seed=7)
    A = mpcf.random.random_pcf((5, 4), generator=gen1)

    gen2 = mpcf.random.Generator(seed=7)
    B = mpcf.random.random_pcf((5, 4), generator=gen2)

    assert A.array_equal(B)


def test_random_pcf_pdist():
    gen = mpcf.random.Generator(seed=8)
    A = mpcf.random.random_pcf((10,), n_range=(10, 100), generator=gen)
    D = mpcf.pdist(A)
    n = 10
    for i in range(n):
        for j in range(i + 1, n):
            d = D[i, j]
            assert np.isfinite(d), f"D[{i},{j}] = {d}"
            assert d >= 0, f"D[{i},{j}] = {d}"


def test_random_pcf_fixed_alpha():
    gen = mpcf.random.Generator(seed=9)
    A = mpcf.random.random_pcf((10,), alpha=2.0, generator=gen)
    for i in range(10):
        arr = A[i].to_numpy()
        assert arr[0, 0] == 0.0
        assert arr[-1, 1] == 0.0


def test_random_pcf_pcf64():
    gen = mpcf.random.Generator(seed=10)
    A = mpcf.random.random_pcf((5,), dtype=mpcf.pcf64, generator=gen)
    assert A.shape == (5,)
    for i in range(5):
        arr = A[i].to_numpy()
        assert arr.dtype == np.float64
        assert arr[-1, 1] == 0.0
