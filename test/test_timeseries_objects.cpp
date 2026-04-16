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

#include <mpcf/functional/pcf.hpp>
#include <mpcf/persistence/barcode.hpp>
#include <mpcf/timeseries.hpp>
#include <mpcf/timeseries/interpolation.hpp>

#include <memory>

namespace
{

using Pcf32 = mpcf::Pcf<mpcf::float32_t, mpcf::float32_t>;
using Barcode32 = mpcf::ph::Barcode<mpcf::float32_t>;
using TsPcf = mpcf::TimeSeries<mpcf::float32_t, Pcf32>;

Pcf32 make_pcf(std::vector<std::pair<float, float>> pts)
{
  std::vector<mpcf::TimePoint<float, float>> points;
  points.reserve(pts.size());
  for (auto& p : pts)
  {
    mpcf::TimePoint<float, float> tp;
    tp.t = p.first;
    tp.v = p.second;
    points.push_back(tp);
  }
  return Pcf32(std::move(points));
}

// ============================================================================
// PCF-valued TimeSeries — C++-only invariants
// Evaluate/construct/linear-blend behaviour is covered by the Python tests in
// test_timeseries_objects.py; those exercise the same C++ evaluate_batch path.
// ============================================================================

TEST(TimeSeriesPcf, RequiresNChannelsOne)
{
  auto p1 = make_pcf({{0.0f, 1.0f}});
  auto p2 = make_pcf({{0.0f, 2.0f}});
  auto p3 = make_pcf({{0.0f, 3.0f}});
  auto p4 = make_pcf({{0.0f, 4.0f}});

  EXPECT_THROW(
      (TsPcf({0.0f, 1.0f}, {p1, p2, p3, p4}, 2, 0.0f, 1.0f)),
      std::invalid_argument);
}

// ============================================================================
// LinearlyBlendable concept (compile-time checks)
// ============================================================================

TEST(InterpolationConcept, ScalarIsBlendable)
{
  static_assert(mpcf::LinearlyBlendable<float, float>);
  static_assert(mpcf::LinearlyBlendable<double, double>);
}

TEST(InterpolationConcept, PcfIsBlendable)
{
  static_assert(mpcf::LinearlyBlendable<float, Pcf32>);
}

TEST(InterpolationConcept, BarcodeIsNotBlendable)
{
  static_assert(!mpcf::LinearlyBlendable<float, Barcode32>);
}

// ============================================================================
// OutOfDomainValue trait
// ============================================================================

TEST(TimeSeriesPcf, OutOfDomainReturnsDefaultPcf)
{
  auto p1 = make_pcf({{0.0f, 1.0f}, {1.0f, 0.0f}});
  auto p2 = make_pcf({{0.0f, 2.0f}, {1.0f, 0.0f}});
  TsPcf ts({0.0f, 1.0f}, {p1, p2}, 1, 0.0f, 1.0f);

  // real_t = -1 is before start_time; evaluator returns OutOfDomainValue
  auto evaluated = ts.evaluate(-1.0f);
  auto pcf = evaluated(std::vector<size_t>{0});
  // Default-constructed Pcf is empty — just verify no crash / no exception
  (void)pcf;
  SUCCEED();
}

// ============================================================================
// Strategy reset on set_interpolation
// ============================================================================

namespace
{
class IdentityStrategy : public mpcf::InterpolationStrategy<float, Pcf32>
{
public:
  std::vector<Pcf32> evaluate(
      const std::vector<float>& /*queries*/,
      const std::vector<float>& /*t_lefts*/,
      const std::vector<float>& /*t_rights*/,
      const std::vector<Pcf32>& v_lefts,
      const std::vector<Pcf32>& /*v_rights*/) const override
  {
    return v_lefts;
  }
};
}

TEST(TimeSeriesPcf, SetInterpolationClearsCustomStrategy)
{
  auto p1 = make_pcf({{0.0f, 1.0f}, {1.0f, 0.0f}});
  auto p2 = make_pcf({{0.0f, 5.0f}, {1.0f, 0.0f}});
  TsPcf ts({0.0f, 1.0f}, {p1, p2}, 1, 0.0f, 1.0f);

  ts.set_strategy(std::make_shared<IdentityStrategy>());
  EXPECT_TRUE(ts.has_custom_strategy());

  ts.set_interpolation(mpcf::InterpolationMode::Nearest);
  EXPECT_FALSE(ts.has_custom_strategy());
}

TEST(TimeSeriesBarcode, SetLinearInterpolationThrows)
{
  mpcf::TimeSeries<float, Barcode32> ts;
  EXPECT_THROW(ts.set_interpolation(mpcf::InterpolationMode::Linear),
               std::invalid_argument);
}

// ============================================================================
// Tensor-valued TimeSeries: Linear explicitly disabled via trait
// ============================================================================

TEST(TimeSeriesPointCloud, SetLinearInterpolationThrows)
{
  // PointCloud<T> = Tensor<T> satisfies LinearlyBlendable syntactically,
  // but blending two tensors with differing shapes is undefined. The
  // disables_linear_interpolation trait suppresses the Linear path.
  mpcf::TimeSeries<float, mpcf::PointCloud<float>> ts;
  EXPECT_THROW(ts.set_interpolation(mpcf::InterpolationMode::Linear),
               std::invalid_argument);
}

TEST(TimeSeriesBctensor, SetLinearInterpolationThrows)
{
  mpcf::TimeSeries<float, mpcf::Tensor<Barcode32>> ts;
  EXPECT_THROW(ts.set_interpolation(mpcf::InterpolationMode::Linear),
               std::invalid_argument);
}

}  // namespace
