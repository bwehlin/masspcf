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
#include <mpcf/algorithms/subdivide.hpp>
#include <mpcf/algorithms/functional/reduce.hpp>
#include <mpcf/algorithms/functional/apply_functional.hpp>
#include <mpcf/algorithms/functional/lp_distance.hpp>
#include <mpcf/algorithms/tensor_eval.hpp>
#include <mpcf/tensor.hpp>

#include <cmath>
#include <vector>

namespace
{

  // ============================================================================
  // subdivide
  // ============================================================================

  TEST(Subdivide, SingleBlock)
  {
    auto blocks = mpcf::subdivide(100, 5);
    ASSERT_EQ(blocks.size(), 1u);
    EXPECT_EQ(blocks[0].first, 0u);
    EXPECT_EQ(blocks[0].second, 4u);
  }

  TEST(Subdivide, ExactMultiple)
  {
    // 10 items, block size 5 => 2 blocks
    auto blocks = mpcf::subdivide(5, 10);
    ASSERT_EQ(blocks.size(), 2u);
    EXPECT_EQ(blocks[0].first, 0u);
    EXPECT_EQ(blocks[0].second, 4u);
    EXPECT_EQ(blocks[1].first, 5u);
    EXPECT_EQ(blocks[1].second, 9u);
  }

  TEST(Subdivide, NonExactMultiple)
  {
    // 7 items, block size 3 => 3 blocks: [0,2], [3,5], [6,6]
    auto blocks = mpcf::subdivide(3, 7);
    ASSERT_EQ(blocks.size(), 3u);
    EXPECT_EQ(blocks[0], std::make_pair(size_t(0), size_t(2)));
    EXPECT_EQ(blocks[1], std::make_pair(size_t(3), size_t(5)));
    EXPECT_EQ(blocks[2], std::make_pair(size_t(6), size_t(6)));
  }

  TEST(Subdivide, BlockSizeOne)
  {
    auto blocks = mpcf::subdivide(1, 4);
    ASSERT_EQ(blocks.size(), 4u);
    for (size_t i = 0; i < 4; ++i)
    {
      EXPECT_EQ(blocks[i].first, i);
      EXPECT_EQ(blocks[i].second, i);
    }
  }

  TEST(Subdivide, SingleItem)
  {
    auto blocks = mpcf::subdivide(5, 1);
    ASSERT_EQ(blocks.size(), 1u);
    EXPECT_EQ(blocks[0].first, 0u);
    EXPECT_EQ(blocks[0].second, 0u);
  }

  TEST(Subdivide, CoverageComplete)
  {
    // Verify all items are covered for various sizes
    for (size_t n = 1; n <= 20; ++n)
    {
      for (size_t bs = 1; bs <= n + 5; ++bs)
      {
        auto blocks = mpcf::subdivide(bs, n);
        EXPECT_EQ(blocks.front().first, 0u);
        EXPECT_EQ(blocks.back().second, n - 1);
        // No gaps between blocks
        for (size_t i = 1; i < blocks.size(); ++i)
        {
          EXPECT_EQ(blocks[i].first, blocks[i - 1].second + 1);
        }
      }
    }
  }

  // ============================================================================
  // lp_distance (C++ level)
  // ============================================================================

  using PcfFloatTypes = ::testing::Types<mpcf::Pcf_f32, mpcf::Pcf_f64>;

  template <typename PcfT>
  class LpDistanceTest : public ::testing::Test {};

  TYPED_TEST_SUITE(LpDistanceTest, PcfFloatTypes);

  TYPED_TEST(LpDistanceTest, IdenticalFunctionsGiveZero)
  {
    using T = typename TypeParam::value_type;
    TypeParam f({{ T(0), T(3) }, { T(1), T(1) }, { T(2), T(0) }});
    auto d = mpcf::lp_distance(f, f);
    EXPECT_NEAR(d, T(0), T(1e-9));
  }

  TYPED_TEST(LpDistanceTest, L1DistanceBasic)
  {
    using T = typename TypeParam::value_type;
    // f = 5 on [0,1), g = 0 => L1 = 5
    TypeParam f({{ T(0), T(5) }, { T(1), T(0) }});
    TypeParam g({{ T(0), T(0) }});
    auto d = mpcf::lp_distance(f, g, T(1));
    EXPECT_NEAR(d, T(5), T(1e-6));
  }

  TYPED_TEST(LpDistanceTest, L2DistanceBasic)
  {
    using T = typename TypeParam::value_type;
    // f = 3 on [0,1), g = 0 => L2 = sqrt(9) = 3
    TypeParam f({{ T(0), T(3) }, { T(1), T(0) }});
    TypeParam g({{ T(0), T(0) }, { T(1), T(0) }});
    auto d = mpcf::lp_distance(f, g, T(2));
    EXPECT_NEAR(d, T(3), T(1e-6));
  }

  TYPED_TEST(LpDistanceTest, SymmetricProperty)
  {
    using T = typename TypeParam::value_type;
    TypeParam f({{ T(0), T(3) }, { T(1), T(1) }, { T(2), T(0) }});
    TypeParam g({{ T(0), T(1) }, { T(2), T(0) }});
    auto d_fg = mpcf::lp_distance(f, g);
    auto d_gf = mpcf::lp_distance(g, f);
    EXPECT_NEAR(d_fg, d_gf, T(1e-9));
  }

  TYPED_TEST(LpDistanceTest, ZeroFunctionsGiveZero)
  {
    using T = typename TypeParam::value_type;
    TypeParam f({{ T(0), T(0) }});
    TypeParam g({{ T(0), T(0) }});
    EXPECT_EQ(mpcf::lp_distance(f, g), T(0));
  }

  TYPED_TEST(LpDistanceTest, NegativeValues)
  {
    using T = typename TypeParam::value_type;
    // f = -3 on [0,1), g = 2 on [0,1) => L1 = |-3-2|*1 = 5
    TypeParam f({{ T(0), T(-3) }, { T(1), T(0) }});
    TypeParam g({{ T(0), T(2) }, { T(1), T(0) }});
    EXPECT_NEAR(mpcf::lp_distance(f, g), T(5), T(1e-6));
  }

  // ============================================================================
  // reduce: combine and parallel_reduce
  // ============================================================================

  template <typename PcfT>
  class ReduceTest : public ::testing::Test {};

  TYPED_TEST_SUITE(ReduceTest, PcfFloatTypes);

  TYPED_TEST(ReduceTest, CombineSumTwoFunctions)
  {
    using PcfT = TypeParam;
    using T = typename PcfT::value_type;

    PcfT f({{ T(0), T(1) }, { T(1), T(0) }});
    PcfT g({{ T(0), T(2) }, { T(1), T(0) }});

    auto sum_op = [](const typename PcfT::rectangle_type& rect) -> T {
      return rect.top + rect.bottom;
    };

    PcfT result = mpcf::combine(f, g, sum_op);
    // f+g = 3 on [0,1), 0 on [1, inf)
    ASSERT_GE(result.size(), 1u);
    EXPECT_EQ(result.points()[0].v, T(3));
  }

  TYPED_TEST(ReduceTest, ReduceMultipleFunctions)
  {
    using PcfT = TypeParam;
    using T = typename PcfT::value_type;

    std::vector<PcfT> fs;
    fs.push_back(PcfT({{ T(0), T(1) }, { T(1), T(0) }}));
    fs.push_back(PcfT({{ T(0), T(2) }, { T(1), T(0) }}));
    fs.push_back(PcfT({{ T(0), T(3) }, { T(1), T(0) }}));

    auto sum_op = [](const typename PcfT::rectangle_type& rect) -> T {
      return rect.top + rect.bottom;
    };

    PcfT result = mpcf::reduce(fs, sum_op);
    // Sum: 1+2+3 = 6 on [0,1)
    EXPECT_EQ(result.points()[0].v, T(6));
  }

  TYPED_TEST(ReduceTest, ParallelReduceMatchesSequentialReduce)
  {
    using PcfT = TypeParam;
    using T = typename PcfT::value_type;

    std::vector<PcfT> fs;
    for (int i = 0; i < 20; ++i)
    {
      fs.push_back(PcfT({{ T(0), T(i + 1) }, { T(1), T(0) }}));
    }

    auto sum_op = [](const typename PcfT::rectangle_type& rect) -> T {
      return rect.top + rect.bottom;
    };

    PcfT seq = mpcf::reduce(fs, sum_op);
    PcfT par = mpcf::parallel_reduce(fs.begin(), fs.end(), sum_op);

    EXPECT_EQ(seq, par);
  }

  // ============================================================================
  // tensor_eval: evaluate PCF tensor at a point
  // ============================================================================

  TEST(TensorEval, EvalAtSinglePoint)
  {
    using PcfT = mpcf::Pcf_f64;
    using T = double;

    // 1D tensor of 3 PCFs
    mpcf::Tensor<PcfT> elems({3});
    elems({0}) = PcfT({{ T(0), T(10) }, { T(1), T(0) }});
    elems({1}) = PcfT({{ T(0), T(20) }, { T(2), T(0) }});
    elems({2}) = PcfT({{ T(0), T(5) }, { T(0.5), T(0) }});

    mpcf::Tensor<T> out({3});
    mpcf::tensor_eval<T, T>(elems, T(0.25), out);

    EXPECT_DOUBLE_EQ(out({0}), 10.0);
    EXPECT_DOUBLE_EQ(out({1}), 20.0);
    EXPECT_DOUBLE_EQ(out({2}), 5.0);
  }

  TEST(TensorEval, EvalAt2DTensor)
  {
    using PcfT = mpcf::Pcf_f64;
    using T = double;

    mpcf::Tensor<PcfT> elems({2, 2});
    elems({0, 0}) = PcfT({{ T(0), T(1) }, { T(1), T(0) }});
    elems({0, 1}) = PcfT({{ T(0), T(2) }, { T(1), T(0) }});
    elems({1, 0}) = PcfT({{ T(0), T(3) }, { T(1), T(0) }});
    elems({1, 1}) = PcfT({{ T(0), T(4) }, { T(1), T(0) }});

    mpcf::Tensor<T> out({2, 2});
    mpcf::tensor_eval<T, T>(elems, T(0.5), out);

    EXPECT_DOUBLE_EQ(out({0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(out({0, 1}), 2.0);
    EXPECT_DOUBLE_EQ(out({1, 0}), 3.0);
    EXPECT_DOUBLE_EQ(out({1, 1}), 4.0);
  }

  TEST(TensorEval, EvalAtTensorOfDomainPoints)
  {
    using PcfT = mpcf::Pcf_f64;
    using T = double;

    // Single PCF: f = 10 on [0,1), 5 on [1,2), 0 after
    mpcf::Tensor<PcfT> elems({1});
    elems({0}) = PcfT({{ T(0), T(10) }, { T(1), T(5) }, { T(2), T(0) }});

    // Evaluate at two domain points
    mpcf::Tensor<T> domain({2});
    domain({0}) = T(0.5);
    domain({1}) = T(1.5);

    // Output shape: (1, 2)
    mpcf::Tensor<T> out({1, 2});
    mpcf::tensor_eval<T, T>(elems, domain, out);

    EXPECT_DOUBLE_EQ(out({0, 0}), 10.0);
    EXPECT_DOUBLE_EQ(out({0, 1}), 5.0);
  }

  // ============================================================================
  // apply_functional: L2 norm, Lp norm, Linfinity norm
  // ============================================================================

  template <typename PcfT>
  class ApplyFunctionalNormsTest : public ::testing::Test {};

  TYPED_TEST_SUITE(ApplyFunctionalNormsTest, PcfFloatTypes);

  TYPED_TEST(ApplyFunctionalNormsTest, L2NormViaApplyFunctional)
  {
    using PcfT = TypeParam;
    using T = typename PcfT::value_type;

    // f = 3 on [0,1) => L2 = sqrt(9) = 3
    PcfT f({{ T(0), T(3) }, { T(1), T(0) }});
    std::vector<PcfT> fs{ f };
    std::vector<T> output(1);

    mpcf::apply_functional(fs.begin(), fs.end(), output.begin(), [](const PcfT& v) {
      return mpcf::l2_norm(v);
    });

    EXPECT_NEAR(static_cast<double>(output[0]), 3.0, 1e-6);
  }

  TYPED_TEST(ApplyFunctionalNormsTest, LpNormViaApplyFunctional)
  {
    using PcfT = TypeParam;
    using T = typename PcfT::value_type;

    // f = 2 on [0,1) => Lp for any p = 2
    PcfT f({{ T(0), T(2) }, { T(1), T(0) }});
    std::vector<PcfT> fs{ f };
    std::vector<T> output(1);

    mpcf::apply_functional(fs.begin(), fs.end(), output.begin(), [](const PcfT& v) {
      return mpcf::lp_norm(v, T(3));
    });

    EXPECT_NEAR(static_cast<double>(output[0]), 2.0, 1e-5);
  }

  TYPED_TEST(ApplyFunctionalNormsTest, LinfinityNormViaApplyFunctional)
  {
    using PcfT = TypeParam;
    using T = typename PcfT::value_type;

    // f = 1 on [0,1), 5 on [1,2), 2 on [2,3) => Linf = 5
    PcfT f({{ T(0), T(1) }, { T(1), T(5) }, { T(2), T(2) }, { T(3), T(0) }});
    std::vector<PcfT> fs{ f };
    std::vector<T> output(1);

    mpcf::apply_functional(fs.begin(), fs.end(), output.begin(), [](const PcfT& v) {
      return mpcf::linfinity_norm(v);
    });

    EXPECT_EQ(output[0], T(5));
  }

  TYPED_TEST(ApplyFunctionalNormsTest, ZeroPcfNormsAreZero)
  {
    using PcfT = TypeParam;
    using T = typename PcfT::value_type;

    PcfT f({{ T(0), T(0) }});
    std::vector<PcfT> fs{ f };
    std::vector<T> output(1);

    mpcf::apply_functional(fs.begin(), fs.end(), output.begin(), [](const PcfT& v) {
      return mpcf::l1_norm(v);
    });
    EXPECT_EQ(output[0], T(0));

    mpcf::apply_functional(fs.begin(), fs.end(), output.begin(), [](const PcfT& v) {
      return mpcf::l2_norm(v);
    });
    EXPECT_EQ(output[0], T(0));

    mpcf::apply_functional(fs.begin(), fs.end(), output.begin(), [](const PcfT& v) {
      return mpcf::linfinity_norm(v);
    });
    EXPECT_EQ(output[0], T(0));
  }

} // namespace
