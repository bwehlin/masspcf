// Copyright 2024-2026 Bjorn Wehlin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
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

#include <mpcf/tensor.hpp>
#include <mpcf/persistence/barcode.hpp>
#include <mpcf/persistence/persistence_pair.hpp>
#include <mpcf/persistence/filter_significant.hpp>
#include <mpcf/executor.hpp>

namespace
{
  using ScalarTypes = ::testing::Types<mpcf::float32_t, mpcf::float64_t>;

  template<typename T>
  class FilterSignificantTest : public ::testing::Test
  {
  };

  TYPED_TEST_SUITE(FilterSignificantTest, ScalarTypes);

  // ============================================================================
  // Empty barcode
  // ============================================================================

  TYPED_TEST(FilterSignificantTest, EmptyBarcodeReturnsEmpty)
  {
    using T = TypeParam;
    using Barcode = mpcf::ph::Barcode<T>;

    Barcode empty;
    auto result = mpcf::ph::filter_significant_bars(empty);
    EXPECT_EQ(result.bars().size(), 0u);
  }

  // ============================================================================
  // Single bar with birth=0 is always kept
  // ============================================================================

  TYPED_TEST(FilterSignificantTest, SingleBarBirthZeroAlwaysKept)
  {
    using T = TypeParam;
    using Pair = mpcf::ph::PersistencePair<T>;
    using Barcode = mpcf::ph::Barcode<T>;

    // birth=0 gives pi=inf, always signal
    std::vector<Pair> bars{ Pair(T(0), T(5)) };
    Barcode bc(std::move(bars));

    auto result = mpcf::ph::filter_significant_bars(bc);
    ASSERT_EQ(result.bars().size(), 1u);
    EXPECT_EQ(result.bars()[0].birth, T(0));
    EXPECT_EQ(result.bars()[0].death, T(5));
  }

  // ============================================================================
  // Bars with infinite death are always kept
  // ============================================================================

  TYPED_TEST(FilterSignificantTest, InfiniteDeathBarsAlwaysKept)
  {
    using T = TypeParam;
    using Pair = mpcf::ph::PersistencePair<T>;
    using Barcode = mpcf::ph::Barcode<T>;

    std::vector<Pair> bars{
      Pair(T(1), std::numeric_limits<T>::infinity()),
      Pair(T(1), T(1.01)),  // noise
      Pair(T(2), T(2.01)),  // noise
    };
    Barcode bc(std::move(bars));

    auto result = mpcf::ph::filter_significant_bars(bc);

    // The infinite bar should always be kept
    bool has_infinite = false;
    for (const auto& bar : result.bars())
    {
      if (Barcode::is_infinite(bar.death))
      {
        has_infinite = true;
      }
    }
    EXPECT_TRUE(has_infinite);
  }

  // ============================================================================
  // Clear signal among noise
  // ============================================================================

  TYPED_TEST(FilterSignificantTest, ClearSignalAmongNoise)
  {
    using T = TypeParam;
    using Pair = mpcf::ph::PersistencePair<T>;
    using Barcode = mpcf::ph::Barcode<T>;

    // One bar with very large pi = 100/1 = 100, rest are noise with pi ~ 1.01
    std::vector<Pair> bars;
    bars.emplace_back(T(1), T(100));  // signal: pi = 100

    for (int i = 0; i < 50; ++i)
    {
      T birth = T(1) + T(i) * T(0.1);
      bars.emplace_back(birth, birth * T(1.01));  // noise: pi = 1.01
    }
    Barcode bc(std::move(bars));

    auto result = mpcf::ph::filter_significant_bars(bc, T(0.05));

    // The signal bar should survive
    ASSERT_GE(result.bars().size(), 1u);

    bool found_signal = false;
    for (const auto& bar : result.bars())
    {
      if (bar.birth == T(1) && bar.death == T(100))
      {
        found_signal = true;
      }
    }
    EXPECT_TRUE(found_signal);

    // Most noise bars should be removed
    EXPECT_LT(result.bars().size(), 10u);
  }

  // ============================================================================
  // All-noise barcode: similar small pi values
  // ============================================================================

  TYPED_TEST(FilterSignificantTest, AllNoiseBarcodeFiltersAll)
  {
    using T = TypeParam;
    using Pair = mpcf::ph::PersistencePair<T>;
    using Barcode = mpcf::ph::Barcode<T>;

    // Many bars with pi ~ 1.05, none significantly different
    std::vector<Pair> bars;
    for (int i = 0; i < 30; ++i)
    {
      T birth = T(1) + T(i) * T(0.1);
      bars.emplace_back(birth, birth * T(1.05));
    }
    Barcode bc(std::move(bars));

    auto result = mpcf::ph::filter_significant_bars(bc, T(0.05));

    // All bars are noise -- none should survive
    EXPECT_EQ(result.bars().size(), 0u);
  }

  // ============================================================================
  // Alpha=1.0 keeps all testable bars
  // ============================================================================

  TYPED_TEST(FilterSignificantTest, HigherAlphaKeepsMore)
  {
    using T = TypeParam;
    using Pair = mpcf::ph::PersistencePair<T>;
    using Barcode = mpcf::ph::Barcode<T>;

    // One signal bar among noise
    std::vector<Pair> bars;
    bars.emplace_back(T(1), T(50));  // signal
    for (int i = 0; i < 20; ++i)
    {
      T birth = T(1) + T(i) * T(0.1);
      bars.emplace_back(birth, birth * T(1.02));
    }
    Barcode bc(std::move(bars));

    auto strict = mpcf::ph::filter_significant_bars(bc, T(0.001));
    auto relaxed = mpcf::ph::filter_significant_bars(bc, T(0.5));

    // Relaxed alpha should keep at least as many bars as strict
    EXPECT_GE(relaxed.bars().size(), strict.bars().size());
  }

  // ============================================================================
  // Tensor task matches single-barcode results
  // ============================================================================

  TYPED_TEST(FilterSignificantTest, TensorTaskMatchesSingleBarcode)
  {
    using T = TypeParam;
    using Pair = mpcf::ph::PersistencePair<T>;
    using Barcode = mpcf::ph::Barcode<T>;

    mpcf::Tensor<Barcode> barcodes({ 3 });

    // Barcode 0: clear signal
    {
      std::vector<Pair> bars;
      bars.emplace_back(T(1), T(100));
      for (int i = 0; i < 20; ++i)
        bars.emplace_back(T(1) + T(i) * T(0.1), (T(1) + T(i) * T(0.1)) * T(1.01));
      barcodes(0) = Barcode(std::move(bars));
    }

    // Barcode 1: all noise
    {
      std::vector<Pair> bars;
      for (int i = 0; i < 15; ++i)
        bars.emplace_back(T(1) + T(i) * T(0.1), (T(1) + T(i) * T(0.1)) * T(1.03));
      barcodes(1) = Barcode(std::move(bars));
    }

    // Barcode 2: empty
    barcodes(2) = Barcode();

    mpcf::Tensor<Barcode> out;
    T alpha = T(0.05);

    auto task = mpcf::ph::make_filter_significant_task(barcodes, out, alpha);
    task->start_async(mpcf::default_executor()).future().wait();

    ASSERT_EQ(out.shape().size(), 1u);
    ASSERT_EQ(out.shape(0), 3u);

    for (size_t i = 0; i < 3; ++i)
    {
      auto expected = mpcf::ph::filter_significant_bars(barcodes(i), alpha);
      EXPECT_TRUE(out(i).is_isomorphic_to(expected))
          << "Mismatch at index " << i;
    }
  }

} // namespace
