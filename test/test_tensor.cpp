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

// ============================================================================
// Helpers
// ============================================================================

namespace
{

  template<typename T>
  mpcf::Tensor<T> make_sequential(const std::vector<size_t>& shape)
  {
    mpcf::Tensor<T> t(shape);
    size_t n = 0;
    t.walk([&t, &n](const std::vector<size_t>& idx)
    {
      t(idx) = static_cast<T>(n++);
    });
    return t;
  }



// ============================================================================
// operator=(const T& val)
// ============================================================================

  TEST(TensorTpp, AssignScalarFillsAllElements)
  {
    mpcf::Tensor<double> t({ 2, 3 });
    t = 7.0;

    t.walk([&t](const std::vector<size_t>& idx)
    {
      EXPECT_EQ(t(idx), 7.0) << "at " << idx[0] << "," << idx[1];
    });
  }

  TEST(TensorTpp, AssignScalarOverwritesPreviousValues)
  {
    mpcf::Tensor<int> t({ 4 });
    t(0) = 100;
    t(1) = 200;
    t(2) = 300;
    t(3) = 400;
    t = 0;
    for (size_t i = 0; i < 4; ++i)
      EXPECT_EQ(t(i), 0);
  }

// ============================================================================
// operator==  / operator!=  — shape mismatch short-circuit
// ============================================================================

  TEST(TensorTpp, InequalityShapeMismatchReturnsFalse)
  {
    mpcf::Tensor<int> a({ 2, 3 });
    mpcf::Tensor<int> b({ 3, 2 });
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a == b);
  }

  TEST(TensorTpp, EqualityIdenticalTensors)
  {
    auto a = make_sequential<float>({ 3, 4 });
    auto b = make_sequential<float>({ 3, 4 });
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
  }

  TEST(TensorTpp, InequalityOneElementDiffers)
  {
    auto a = make_sequential<double>({ 2, 2 });
    auto b = make_sequential<double>({ 2, 2 });
    b({ 1, 1 }) = 999.0;
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
  }

// ============================================================================
// assign_from
// ============================================================================

  TEST(TensorTpp, AssignFromCopiesValues)
  {
    auto src = make_sequential<double>({ 2, 3 });
    mpcf::Tensor<double> dst({ 2, 3 });
    dst.assign_from(src);
    EXPECT_EQ(dst, src);
  }

  TEST(TensorTpp, AssignFromShapeMismatchThrows)
  {
    mpcf::Tensor<int> src({ 2, 3 });
    mpcf::Tensor<int> dst({ 3, 2 });
    EXPECT_THROW(dst.assign_from(src), std::runtime_error);
  }

// ============================================================================
// size()
// ============================================================================

  TEST(TensorTpp, SizeOfEmptyTensor)
  {
    mpcf::Tensor<int> t;
    // Default constructor → shape {}, size = 0
    EXPECT_EQ(t.size(), 0u);
  }

  TEST(TensorTpp, Size1d)
  {
    mpcf::Tensor<float> t({ 7 });
    EXPECT_EQ(t.size(), 7u);
  }

  TEST(TensorTpp, Size3d)
  {
    mpcf::Tensor<double> t({ 2, 3, 5 });
    EXPECT_EQ(t.size(), 30u);
  }

// ============================================================================
// copy()
// ============================================================================

  TEST(TensorTpp, CopyIsDeepCopy)
  {
    auto original = make_sequential<int>({ 3, 3 });
    auto copy = original.copy();

    EXPECT_EQ(original, copy);

    copy({ 0, 0 }) = 9999;
    EXPECT_NE(original({ 0, 0 }), copy({ 0, 0 }));
  }

  TEST(TensorTpp, CopyOfNonContiguousTensorIsContiguous)
  {
    auto t = make_sequential<double>({ 4, 4 });
    auto sliced = t[std::vector<mpcf::Slice>{ mpcf::range(1, 3, std::nullopt), mpcf::all() }];
    EXPECT_FALSE(sliced.is_contiguous());

    auto c = sliced.copy();
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_EQ(c.shape(0), 2u);
    EXPECT_EQ(c.shape(1), 4u);

    // Values should match the slice
    for (size_t i = 0; i < 2; ++i)
      for (size_t j = 0; j < 4; ++j)
        EXPECT_EQ(c({ i, j }), sliced({ i, j }));
  }

// ============================================================================
// flatten()
// ============================================================================

  TEST(TensorTpp, FlattenContiguous2d)
  {
    auto t = make_sequential<int>({ 2, 3 });
    auto flat = t.flatten();

    ASSERT_EQ(flat.shape().size(), 1u);
    EXPECT_EQ(flat.shape(0), 6u);

    for (size_t i = 0; i < 6; ++i)
      EXPECT_EQ(flat(i), static_cast<int>(i));
  }

  TEST(TensorTpp, FlattenNonContiguousWorksOnCopy)
  {
    auto t = make_sequential<double>({ 4, 4 });
    auto sliced = t[std::vector<mpcf::Slice>{ mpcf::range(0, 4, 2), mpcf::all() }];
    EXPECT_FALSE(sliced.is_contiguous());
    auto flat = sliced.flatten();
    EXPECT_TRUE(flat.is_contiguous());

    EXPECT_NE(sliced.data(), flat.data());
  }

// ============================================================================
// walk() — bool-returning functor (early exit)
// ============================================================================

  TEST(TensorTpp, WalkBoolFunctorStopsEarly)
  {
    auto t = make_sequential<int>({ 10 });
    int count = 0;
    t.walk([&count](const std::vector<size_t>& /*idx*/) -> bool
    {
      ++count;
      return count < 5; // stop after 5 visits
    });
    EXPECT_EQ(count, 5);
  }

  TEST(TensorTpp, WalkBoolFunctorAllTrue)
  {
    auto t = make_sequential<int>({ 4 });
    int count = 0;
    t.walk([&count](const std::vector<size_t>&) -> bool
    {
      ++count;
      return true;
    });
    EXPECT_EQ(count, 4);
  }

  TEST(TensorTpp, WalkEmptyShapeDoesNothing)
  {
    mpcf::Tensor<int> t; // shape {}
    int count = 0;
    // shape is empty → walk returns immediately
    t.walk([&count](const std::vector<size_t>&) { ++count; });
    // shape {} has no zero dimension, so walk does visit the one scalar element
    // (walk only returns early if shape is empty OR any dimension is 0)
    EXPECT_EQ(count, 0);
  }

  TEST(TensorTpp, WalkZeroDimensionDoesNothing)
  {
    mpcf::Tensor<int> t({ 0, 3 });
    int count = 0;
    t.walk([&count](const std::vector<size_t>&) { ++count; });
    EXPECT_EQ(count, 0);
  }

// ============================================================================
// apply()
// ============================================================================

  TEST(TensorTpp, ApplyDoubleAllElements)
  {
    auto t = make_sequential<double>({ 3, 3 });
    t.apply([](double& v) { v *= 2.0; });

    size_t n = 0;
    t.walk([&t, &n](const std::vector<size_t>& idx)
    {
      EXPECT_EQ(t(idx), static_cast<double>(n) * 2.0);
      ++n;
    });
  }

  TEST(TensorTpp, ApplyOnEmptyDimensionDoesNothing)
  {
    mpcf::Tensor<int> t({ 0 });
    int calls = 0;
    t.apply([&calls](int&) { ++calls; });
    EXPECT_EQ(calls, 0);
  }

// ============================================================================
// extract() / operator[] — various Slice types
// ============================================================================

  TEST(TensorTpp, SliceAllPreservesShape)
  {
    auto t = make_sequential<int>({ 3, 4 });
    auto view = t[std::vector<mpcf::Slice>{ mpcf::all(), mpcf::all() }];
    EXPECT_EQ(view.shape(0), 3u);
    EXPECT_EQ(view.shape(1), 4u);
  }

  TEST(TensorTpp, SliceIndexDropsDimension)
  {
    auto t = make_sequential<int>({ 3, 4 });
    // Index into first dim → shape becomes (4,)
    auto view = t[std::vector<mpcf::Slice>{ mpcf::index(1), mpcf::all() }];
    EXPECT_EQ(view.shape().size(), 1u);
    EXPECT_EQ(view.shape(0), 4u);
    // Row 1: values 4..7
    for (size_t j = 0; j < 4; ++j)
      EXPECT_EQ(view({ j }), static_cast<int>(4 + j));
  }

  TEST(TensorTpp, SliceRangeStopClampedToShape)
  {
    auto t = make_sequential<int>({ 5 });
    // stop=100 should be clamped to 5
    auto view = t[std::vector<mpcf::Slice>{ mpcf::range(1, 100, std::nullopt) }];
    EXPECT_EQ(view.shape(0), 4u);
  }

  TEST(TensorTpp, SliceRangeStartNegativeClampsToZero)
  {
    auto t = make_sequential<int>({ 5 });
    auto view = t[std::vector<mpcf::Slice>{ mpcf::range(-10, 3, std::nullopt) }];
    // start clamped to 0, stop=3 → size 3
    EXPECT_EQ(view.shape(0), 3u);
    EXPECT_EQ(view({ 0 }), 0);
    EXPECT_EQ(view({ 1 }), 1);
    EXPECT_EQ(view({ 2 }), 2);
  }

  TEST(TensorTpp, SliceRangeStopLessThanStartGivesZeroSize)
  {
    auto t = make_sequential<int>({ 5 });
    auto view = t[std::vector<mpcf::Slice>{ mpcf::range(3, 1, std::nullopt) }];
    EXPECT_EQ(view.shape(0), 0u);
  }

  TEST(TensorTpp, SliceRangeZeroStepGivesZeroSize)
  {
    auto t = make_sequential<int>({ 5 });
    auto view = t[std::vector<mpcf::Slice>{ mpcf::range(0, 5, 0) }];
    EXPECT_EQ(view.shape(0), 0u);
  }

  TEST(TensorTpp, SliceRangeNegativeStepThrows)
  {
    auto t = make_sequential<int>({ 5 });
    EXPECT_THROW(
        (t[std::vector<mpcf::Slice>{ mpcf::range(4, 0, -1) }]),
        std::runtime_error
    );
  }

  TEST(TensorTpp, SliceRangeWithDefaultStartStop)
  {
    auto t = make_sequential<int>({ 5 });
    // range with no start/stop → full range
    auto view = t[std::vector<mpcf::Slice>{ mpcf::range(std::nullopt, std::nullopt, std::nullopt) }];
    EXPECT_EQ(view.shape(0), 5u);
    for (size_t i = 0; i < 5; ++i)
      EXPECT_EQ(view({ i }), static_cast<int>(i));
  }

  TEST(TensorTpp, Slice3dMixed)
  {
    auto t = make_sequential<int>({ 4, 5, 6 });
    // t[1, 2:4, ::2]
    auto view = t[std::vector<mpcf::Slice>{
        mpcf::index(1),
        mpcf::range(2, 4, std::nullopt),
        mpcf::range(std::nullopt, std::nullopt, 2)
    }];

    EXPECT_EQ(view.shape().size(), 2u);
    EXPECT_EQ(view.shape(0), 2u);   // rows 2,3
    EXPECT_EQ(view.shape(1), 3u);   // cols 0,2,4

    // t[1,2,0]=1*30+2*6+0=42; t[1,2,2]=44; t[1,2,4]=46
    EXPECT_EQ(view({ 0, 0 }), 1 * 30 + 2 * 6 + 0);
    EXPECT_EQ(view({ 0, 1 }), 1 * 30 + 2 * 6 + 2);
    EXPECT_EQ(view({ 0, 2 }), 1 * 30 + 2 * 6 + 4);
    EXPECT_EQ(view({ 1, 0 }), 1 * 30 + 3 * 6 + 0);
  }

// ============================================================================
// Flattened view index_to_data_index: non-1d index throws
// ============================================================================

  TEST(TensorTpp, FlattenedViewNon1dIndexThrows)
  {
    auto t = make_sequential<int>({ 2, 3 });
    auto flat = t.flatten();
    // Accessing flat via a 2d index should throw
    EXPECT_THROW((void)flat({ 0, 0 }), std::runtime_error);
  }

// ============================================================================
// 1d operator()(size_t) convenience overload
// ============================================================================

  TEST(TensorTpp, SingleIndexOverload1d)
  {
    mpcf::Tensor<int> t({ 5 });
    for (size_t i = 0; i < 5; ++i)
      t(i) = static_cast<int>(i * 10);
    for (size_t i = 0; i < 5; ++i)
      EXPECT_EQ(t(i), static_cast<int>(i * 10));
  }

} // namespace