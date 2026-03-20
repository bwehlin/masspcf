// Copyright 2024-2026 Bjorn Wehlin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <mpcf/tensor.h>

// Minimal C++ smoke tests for masked operations.
// Full coverage lives in test/python/test_tensor_mask.py.

namespace
{

  TEST(MaskedSelect, Smoke)
  {
    mpcf::Tensor<float> t({4});
    t({0}) = 10.0f; t({1}) = 20.0f; t({2}) = 30.0f; t({3}) = 40.0f;

    mpcf::Tensor<bool> mask({4});
    mask({0}) = true; mask({1}) = false; mask({2}) = true; mask({3}) = false;

    auto result = mpcf::masked_select(t, mask);
    ASSERT_EQ(result.shape(), (std::vector<size_t>{2}));
    EXPECT_FLOAT_EQ(result({0}), 10.0f);
    EXPECT_FLOAT_EQ(result({1}), 30.0f);
  }

  TEST(MaskedAssign, Smoke)
  {
    mpcf::Tensor<float> t({3}, 0.0f);
    mpcf::Tensor<bool> mask({3});
    mask({0}) = false; mask({1}) = true; mask({2}) = false;

    mpcf::Tensor<float> values({1});
    values({0}) = 99.0f;

    mpcf::masked_assign(t, mask, values);
    EXPECT_FLOAT_EQ(t({0}), 0.0f);
    EXPECT_FLOAT_EQ(t({1}), 99.0f);
    EXPECT_FLOAT_EQ(t({2}), 0.0f);
  }

  TEST(MaskedFill, Smoke)
  {
    mpcf::Tensor<float> t({3});
    t({0}) = 1.0f; t({1}) = 2.0f; t({2}) = 3.0f;

    mpcf::Tensor<bool> mask({3}, true);
    mpcf::masked_fill(t, mask, -1.0f);

    EXPECT_FLOAT_EQ(t({0}), -1.0f);
    EXPECT_FLOAT_EQ(t({1}), -1.0f);
    EXPECT_FLOAT_EQ(t({2}), -1.0f);
  }

  TEST(AxisSelect, Smoke2D)
  {
    // 2x3 tensor, select along axis 1 with mask [T, F, T]
    mpcf::Tensor<float> t({2, 3});
    t({0, 0}) = 0; t({0, 1}) = 1; t({0, 2}) = 2;
    t({1, 0}) = 3; t({1, 1}) = 4; t({1, 2}) = 5;

    mpcf::Tensor<bool> mask({3});
    mask({0}) = true; mask({1}) = false; mask({2}) = true;

    auto result = mpcf::axis_select(t, 1, mask);
    ASSERT_EQ(result.shape(), (std::vector<size_t>{2, 2}));
    EXPECT_FLOAT_EQ(result({0, 0}), 0);
    EXPECT_FLOAT_EQ(result({0, 1}), 2);
    EXPECT_FLOAT_EQ(result({1, 0}), 3);
    EXPECT_FLOAT_EQ(result({1, 1}), 5);
  }

} // namespace
