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

#include <mpcf/io.hpp>
#include <mpcf/timeseries.hpp>

#include <sstream>

namespace
{
  template <typename T>
  class TimeSeriesIoTest : public ::testing::Test
  {
  };

  using FloatTypes = ::testing::Types<mpcf::float32_t, mpcf::float64_t>;
  TYPED_TEST_SUITE(TimeSeriesIoTest, FloatTypes);

  // ============================================================================
  // TimeSeries tensor roundtrip
  // ============================================================================

  TYPED_TEST(TimeSeriesIoTest, TensorRoundtrip)
  {
    using Tt = TypeParam;
    using Tv = TypeParam;
    using TsT = mpcf::TimeSeries<Tt, Tv>;
    using TensorT = mpcf::Tensor<TsT>;

    TsT ts1({Tt(0), Tt(1), Tt(2)}, {Tv(10), Tv(20), Tv(30)},
            1, Tt(0), Tt(1));
    TsT ts2({Tt(0), Tt(0.5)}, {Tv(100), Tv(200)},
            1, Tt(5), Tt(0.5));

    TensorT tensor({2});
    tensor(0) = ts1;
    tensor(1) = ts2;

    std::stringstream ss;
    mpcf::write(tensor, ss);

    std::istringstream iss(ss.str());
    auto restored = mpcf::read<TensorT>(iss);

    ASSERT_EQ(restored.shape(), tensor.shape());
    for (size_t i = 0; i < tensor.size(); ++i)
    {
      EXPECT_EQ(restored(i), tensor(i));
    }
  }

  // ============================================================================
  // TimeSeries object roundtrip
  // ============================================================================

  TYPED_TEST(TimeSeriesIoTest, ObjectRoundtrip)
  {
    using Tt = TypeParam;
    using Tv = TypeParam;
    using TsT = mpcf::TimeSeries<Tt, Tv>;

    TsT ts({Tt(0), Tt(1), Tt(2)}, {Tv(10), Tv(20), Tv(30)},
            1, Tt(100), Tt(0.25));

    std::stringstream ss;
    mpcf::write_object(ts, ss);

    std::istringstream iss(ss.str());
    auto result = mpcf::read_any_object(iss);

    auto* restored = std::get_if<TsT>(&result);
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(*restored, ts);
  }

  // ============================================================================
  // Multi-channel TimeSeries roundtrip
  // ============================================================================

  TYPED_TEST(TimeSeriesIoTest, MultiChannelRoundtrip)
  {
    using Tt = TypeParam;
    using Tv = TypeParam;
    using TsT = mpcf::TimeSeries<Tt, Tv>;
    using TensorT = mpcf::Tensor<TsT>;

    // 3 time points, 2 channels -> 6 values
    TsT ts({Tt(0), Tt(1), Tt(2)},
            {Tv(1), Tv(2), Tv(3), Tv(4), Tv(5), Tv(6)},
            2, Tt(0), Tt(1));

    TensorT tensor({1});
    tensor(0) = ts;

    std::stringstream ss;
    mpcf::write(tensor, ss);

    std::istringstream iss(ss.str());
    auto restored = mpcf::read<TensorT>(iss);

    ASSERT_EQ(restored.shape(), tensor.shape());
    EXPECT_EQ(restored(0), tensor(0));
  }

  // ============================================================================
  // Linear interpolation mode preserved
  // ============================================================================

  TYPED_TEST(TimeSeriesIoTest, LinearInterpolationRoundtrip)
  {
    using Tt = TypeParam;
    using Tv = TypeParam;
    using TsT = mpcf::TimeSeries<Tt, Tv>;

    TsT ts({Tt(0), Tt(1), Tt(2)}, {Tv(0), Tv(10), Tv(20)},
            1, Tt(0), Tt(1), mpcf::InterpolationMode::Linear);

    ASSERT_EQ(ts.interpolation(), mpcf::InterpolationMode::Linear);

    std::stringstream ss;
    mpcf::write_object(ts, ss);

    std::istringstream iss(ss.str());
    auto result = mpcf::read_any_object(iss);

    auto* restored = std::get_if<TsT>(&result);
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(restored->interpolation(), mpcf::InterpolationMode::Linear);
    EXPECT_EQ(*restored, ts);
  }

  // ============================================================================
  // read_any_tensor dispatches correctly for TimeSeries
  // ============================================================================

  TYPED_TEST(TimeSeriesIoTest, ReadAnyTensorDispatch)
  {
    using Tt = TypeParam;
    using Tv = TypeParam;
    using TsT = mpcf::TimeSeries<Tt, Tv>;
    using TensorT = mpcf::Tensor<TsT>;

    TsT ts({Tt(0), Tt(1)}, {Tv(5), Tv(10)},
            1, Tt(0), Tt(1));

    TensorT tensor({1});
    tensor(0) = ts;

    std::stringstream ss;
    mpcf::write(tensor, ss);

    std::istringstream iss(ss.str());
    auto result = mpcf::read_any_tensor(iss);

    auto* restored = std::get_if<TensorT>(&result);
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ((*restored)(0), ts);
  }

  // ============================================================================
  // Format ID correctness
  // ============================================================================

  TEST(TimeSeriesIoCore, FormatIds)
  {
    {
      auto fmt = mpcf::io::detail::tensorFormat<mpcf::TimeSeries_f32>();
      EXPECT_EQ(20000, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
    }
    {
      auto fmt = mpcf::io::detail::tensorFormat<mpcf::TimeSeries_f64>();
      EXPECT_EQ(20000, fmt.baseFormat);
      EXPECT_EQ(64, fmt.subFormat);
    }
  }

} // namespace
