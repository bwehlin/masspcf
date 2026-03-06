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
// Typed test suite — runs every test body for int, float, and double
// ============================================================================

  using TensorTypes = ::testing::Types<int, float, double>;

  template <typename T>
  class TensorTppTyped : public ::testing::Test {};
  TYPED_TEST_SUITE(TensorTppTyped, TensorTypes);

// operator=(const T& val)

  TYPED_TEST(TensorTppTyped, AssignScalarFillsAllElements)
  {
    using T = TypeParam;
    mpcf::Tensor<T> t({ 2, 3 });
    t = T(7);
    t.walk([&t](const std::vector<size_t>& idx) { EXPECT_EQ(t(idx), T(7)); });
  }

  TYPED_TEST(TensorTppTyped, AssignScalarOverwritesPreviousValues)
  {
    using T = TypeParam;
    mpcf::Tensor<T> t({ 4 });
    for (size_t i = 0; i < 4; ++i) t(i) = T(100 + i);
    t = T(0);
    for (size_t i = 0; i < 4; ++i) EXPECT_EQ(t(i), T(0));
  }

// operator== / operator!=

  TYPED_TEST(TensorTppTyped, EqualityShapeMismatch)
  {
    using T = TypeParam;
    mpcf::Tensor<T> a({ 2, 3 });
    mpcf::Tensor<T> b({ 3, 2 });
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
  }

  TYPED_TEST(TensorTppTyped, EqualityIdentical)
  {
    using T = TypeParam;
    auto a = make_sequential<T>({ 3, 4 });
    auto b = make_sequential<T>({ 3, 4 });
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
  }

  TYPED_TEST(TensorTppTyped, InequalityOneElementDiffers)
  {
    using T = TypeParam;
    auto a = make_sequential<T>({ 2, 2 });
    auto b = make_sequential<T>({ 2, 2 });
    b({ 1, 1 }) = T(999);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
  }

// assign_from

  TYPED_TEST(TensorTppTyped, AssignFromCopiesValues)
  {
    using T = TypeParam;
    auto src = make_sequential<T>({ 2, 3 });
    mpcf::Tensor<T> dst({ 2, 3 });
    dst.assign_from(src);
    EXPECT_EQ(dst, src);
  }

  TYPED_TEST(TensorTppTyped, AssignFromShapeMismatchThrows)
  {
    using T = TypeParam;
    mpcf::Tensor<T> src({ 2, 3 });
    mpcf::Tensor<T> dst({ 3, 2 });
    EXPECT_THROW(dst.assign_from(src), std::runtime_error);
  }

// size()

  TYPED_TEST(TensorTppTyped, SizeOfEmptyTensor)
  {
    using T = TypeParam;
    mpcf::Tensor<T> t;
    EXPECT_EQ(t.size(), 0u);
  }

  TYPED_TEST(TensorTppTyped, Size1d)
  {
    using T = TypeParam;
    mpcf::Tensor<T> t({ 7 });
    EXPECT_EQ(t.size(), 7u);
  }

  TYPED_TEST(TensorTppTyped, Size3d)
  {
    using T = TypeParam;
    mpcf::Tensor<T> t({ 2, 3, 5 });
    EXPECT_EQ(t.size(), 30u);
  }

// copy()

  TYPED_TEST(TensorTppTyped, CopyIsDeepCopy)
  {
    using T = TypeParam;
    auto original = make_sequential<T>({ 3, 3 });
    auto copy = original.copy();
    EXPECT_EQ(original, copy);
    copy({ 0, 0 }) = T(9999);
    EXPECT_NE(original({ 0, 0 }), copy({ 0, 0 }));
  }

  TYPED_TEST(TensorTppTyped, CopyOfNonContiguousTensorIsContiguous)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4, 4 });
    auto sliced = t[std::vector<mpcf::Slice>{ mpcf::range(1, 3, std::nullopt), mpcf::all() }];
    EXPECT_FALSE(sliced.is_contiguous());
    auto c = sliced.copy();
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_EQ(c.shape(0), 2u);
    EXPECT_EQ(c.shape(1), 4u);
    for (size_t i = 0; i < 2; ++i)
      for (size_t j = 0; j < 4; ++j)
        EXPECT_EQ(c({ i, j }), sliced({ i, j }));
  }

// flatten()

  TYPED_TEST(TensorTppTyped, FlattenContiguous)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 2, 3 });
    auto flat = t.flatten();
    ASSERT_EQ(flat.shape().size(), 1u);
    EXPECT_EQ(flat.shape(0), 6u);
    for (size_t i = 0; i < 6; ++i)
      EXPECT_EQ(flat(i), T(i));
  }

  TYPED_TEST(TensorTppTyped, FlattenNonContiguous)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4, 4 });
    auto sliced = t[std::vector<mpcf::Slice>{ mpcf::range(0, 4, 2), mpcf::all() }];
    EXPECT_FALSE(sliced.is_contiguous());
    auto flat = sliced.flatten();
    EXPECT_TRUE(flat.is_contiguous());
    EXPECT_NE(sliced.data(), flat.data());
  }

// walk()

  TYPED_TEST(TensorTppTyped, WalkBoolFunctorStopsEarly)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 10 });
    int count = 0;
    t.walk([&count](const std::vector<size_t>&) -> bool { return ++count < 5; });
    EXPECT_EQ(count, 5);
  }

  TYPED_TEST(TensorTppTyped, WalkBoolFunctorAllTrue)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4 });
    int count = 0;
    t.walk([&count](const std::vector<size_t>&) -> bool { ++count; return true; });
    EXPECT_EQ(count, 4);
  }

  TYPED_TEST(TensorTppTyped, WalkEmptyShapeDoesNothing)
  {
    using T = TypeParam;
    mpcf::Tensor<T> t;
    int count = 0;
    t.walk([&count](const std::vector<size_t>&) { ++count; });
    EXPECT_EQ(count, 0);
  }

  TYPED_TEST(TensorTppTyped, WalkZeroDimensionDoesNothing)
  {
    using T = TypeParam;
    mpcf::Tensor<T> t({ 0, 3 });
    int count = 0;
    t.walk([&count](const std::vector<size_t>&) { ++count; });
    EXPECT_EQ(count, 0);
  }

// apply()

  TYPED_TEST(TensorTppTyped, ApplyMultipliesAllElements)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 3, 3 });
    t.apply([](T& v) { v = v * T(2); });
    size_t n = 0;
    t.walk([&t, &n](const std::vector<size_t>& idx) {
      EXPECT_EQ(t(idx), T(n) * T(2));
      ++n;
    });
  }

  TYPED_TEST(TensorTppTyped, ApplyOnEmptyDimensionDoesNothing)
  {
    using T = TypeParam;
    mpcf::Tensor<T> t({ 0 });
    int calls = 0;
    t.apply([&calls](T&) { ++calls; });
    EXPECT_EQ(calls, 0);
  }

// extract() / operator[]

  TYPED_TEST(TensorTppTyped, SliceAllPreservesShape)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 3, 4 });
    auto view = t[std::vector<mpcf::Slice>{ mpcf::all(), mpcf::all() }];
    EXPECT_EQ(view.shape(0), 3u);
    EXPECT_EQ(view.shape(1), 4u);
  }

  TYPED_TEST(TensorTppTyped, SliceIndexDropsDimension)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 3, 4 });
    auto view = t[std::vector<mpcf::Slice>{ mpcf::index(1), mpcf::all() }];
    EXPECT_EQ(view.shape().size(), 1u);
    EXPECT_EQ(view.shape(0), 4u);
    for (size_t j = 0; j < 4; ++j)
      EXPECT_EQ(view({ j }), T(4 + j));
  }

  TYPED_TEST(TensorTppTyped, SliceRangeStopClampedToShape)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });
    auto view = t[std::vector<mpcf::Slice>{ mpcf::range(1, 100, std::nullopt) }];
    EXPECT_EQ(view.shape(0), 4u);
  }

  TYPED_TEST(TensorTppTyped, SliceRangeStartNegativeClampsToZero)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });
    auto view = t[std::vector<mpcf::Slice>{ mpcf::range(-10, 3, std::nullopt) }];
    EXPECT_EQ(view.shape(0), 3u);
    EXPECT_EQ(view({ 0 }), T(0));
    EXPECT_EQ(view({ 1 }), T(1));
    EXPECT_EQ(view({ 2 }), T(2));
  }

  TYPED_TEST(TensorTppTyped, SliceRangeStopLessThanStartGivesZeroSize)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });
    auto view = t[std::vector<mpcf::Slice>{ mpcf::range(3, 1, std::nullopt) }];
    EXPECT_EQ(view.shape(0), 0u);
  }

  TYPED_TEST(TensorTppTyped, SliceRangeZeroStepGivesZeroSize)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });
    auto view = t[std::vector<mpcf::Slice>{ mpcf::range(0, 5, 0) }];
    EXPECT_EQ(view.shape(0), 0u);
  }

  TYPED_TEST(TensorTppTyped, SliceRangeNegativeStepThrows)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });
    EXPECT_THROW(
        (t[std::vector<mpcf::Slice>{ mpcf::range(4, 0, -1) }]),
        std::runtime_error
    );
  }

  TYPED_TEST(TensorTppTyped, SliceRangeWithDefaultStartStop)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });
    auto view = t[std::vector<mpcf::Slice>{ mpcf::range(std::nullopt, std::nullopt, std::nullopt) }];
    EXPECT_EQ(view.shape(0), 5u);
    for (size_t i = 0; i < 5; ++i)
      EXPECT_EQ(view({ i }), T(i));
  }

  TYPED_TEST(TensorTppTyped, Slice3dMixed)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4, 5, 6 });
    auto view = t[std::vector<mpcf::Slice>{
        mpcf::index(1),
        mpcf::range(2, 4, std::nullopt),
        mpcf::range(std::nullopt, std::nullopt, 2)
    }];
    EXPECT_EQ(view.shape().size(), 2u);
    EXPECT_EQ(view.shape(0), 2u);
    EXPECT_EQ(view.shape(1), 3u);
    EXPECT_EQ(view({ 0, 0 }), T(1 * 30 + 2 * 6 + 0));
    EXPECT_EQ(view({ 0, 1 }), T(1 * 30 + 2 * 6 + 2));
    EXPECT_EQ(view({ 0, 2 }), T(1 * 30 + 2 * 6 + 4));
    EXPECT_EQ(view({ 1, 0 }), T(1 * 30 + 3 * 6 + 0));
  }

// flatten() non-1d index throws

  TYPED_TEST(TensorTppTyped, FlattenedViewNon1dIndexThrows)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 2, 3 });
    auto flat = t.flatten();
    EXPECT_THROW((void)flat({ 0, 0 }), std::runtime_error);
  }

// 1d operator()(size_t) overload

  TYPED_TEST(TensorTppTyped, SingleIndexOverload1d)
  {
    using T = TypeParam;
    mpcf::Tensor<T> t({ 5 });
    for (size_t i = 0; i < 5; ++i) t(i) = T(i * 10);
    for (size_t i = 0; i < 5; ++i) EXPECT_EQ(t(i), T(i * 10));
  }

// any_of / any_of_idx

  TYPED_TEST(TensorTppTyped, AnyOfReturnsTrueWhenPredicateMatches)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4 });
    EXPECT_TRUE(t.any_of([](const T& v) { return v == T(3); }));
  }

  TYPED_TEST(TensorTppTyped, AnyOfReturnsFalseWhenNoMatch)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4 });
    EXPECT_FALSE(t.any_of([](const T& v) { return v > T(100); }));
  }

  TYPED_TEST(TensorTppTyped, AnyOfIdxReturnsTrueWhenIndexMatches)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 3, 3 });
    EXPECT_TRUE(t.any_of_idx([&t](const std::vector<size_t>& idx) { return t(idx) == T(8); }));
  }

  TYPED_TEST(TensorTppTyped, AnyOfIdxReturnsFalseWhenNoMatch)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 3, 3 });
    EXPECT_FALSE(t.any_of_idx([&t](const std::vector<size_t>& idx) { return t(idx) > T(100); }));
  }

// rank() / strides() / stride() / offset()

  TYPED_TEST(TensorTppTyped, RankMatchesDimensionCount)
  {
    using T = TypeParam;
    EXPECT_EQ(mpcf::Tensor<T>({ 5 }).rank(), 1u);
    EXPECT_EQ(mpcf::Tensor<T>({ 3, 4 }).rank(), 2u);
    EXPECT_EQ(mpcf::Tensor<T>({ 2, 3, 5 }).rank(), 3u);
  }

  TYPED_TEST(TensorTppTyped, StridesCorrectForRowMajor2d)
  {
    using T = TypeParam;
    mpcf::Tensor<T> t({ 3, 4 });
    ASSERT_EQ(t.strides().size(), 2u);
    EXPECT_EQ(t.stride(0), 4u);
    EXPECT_EQ(t.stride(1), 1u);
  }

  TYPED_TEST(TensorTppTyped, OffsetNonZeroAfterIndexSlice)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4, 4 });
    auto view = t[std::vector<mpcf::Slice>{ mpcf::index(1), mpcf::all() }];
    EXPECT_EQ(view.offset(), 4u);
  }

// ============================================================================
// Cross-type tests (not parameterizable on a single T)
// ============================================================================

  TEST(TensorTpp, AssignFromCrossTypeIntToDouble)
  {
    mpcf::Tensor<int> src({ 2, 2 });
    src({ 0, 0 }) = 1; src({ 0, 1 }) = 2;
    src({ 1, 0 }) = 3; src({ 1, 1 }) = 4;

    mpcf::Tensor<double> dst({ 2, 2 });
    dst.assign_from(src);

    EXPECT_DOUBLE_EQ(dst({ 0, 0 }), 1.0);
    EXPECT_DOUBLE_EQ(dst({ 0, 1 }), 2.0);
    EXPECT_DOUBLE_EQ(dst({ 1, 0 }), 3.0);
    EXPECT_DOUBLE_EQ(dst({ 1, 1 }), 4.0);
  }

  TEST(TensorTpp, AssignFromCrossTypeShapeMismatchThrows)
  {
    mpcf::Tensor<float> src({ 2, 3 });
    mpcf::Tensor<double> dst({ 3, 2 });
    EXPECT_THROW(dst.assign_from(src), std::runtime_error);
  }

  TEST(TensorTpp, CrossTypeEqualityIntAndDouble)
  {
    mpcf::Tensor<int>    a({ 3 });
    mpcf::Tensor<double> b({ 3 });
    for (size_t i = 0; i < 3; ++i)
    {
      a(i) = static_cast<int>(i);
      b(i) = static_cast<double>(i);
    }
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
  }

  TEST(TensorTpp, CrossTypeInequalityIntAndDouble)
  {
    mpcf::Tensor<int>    a({ 3 });
    mpcf::Tensor<double> b({ 3 });
    for (size_t i = 0; i < 3; ++i)
    {
      a(i) = static_cast<int>(i);
      b(i) = static_cast<double>(i) + 0.5;
    }
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
  }

} // namespace