#include <gtest/gtest.h>

#include <sbear/tensor.hpp>
#include <sbear/walk.hpp>
#include <sbear/distance_matrix.hpp>
#include <sbear/point_cloud.hpp>
#include <sbear/symmetric_matrix.hpp>

// ============================================================================
// Helpers
// ============================================================================

namespace
{

  template<typename T>
  sb::Tensor<T> make_sequential(const std::vector<size_t>& shape)
  {
    sb::Tensor<T> t(shape);
    size_t n = 0;
    sb::walk(t, [&t, &n](const std::vector<size_t>& idx)
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
    sb::Tensor<T> t({ 2, 3 });
    t = T(7);
    sb::walk(t, [&t](const std::vector<size_t>& idx) { EXPECT_EQ(t(idx), T(7)); });
  }

  TYPED_TEST(TensorTppTyped, AssignScalarOverwritesPreviousValues)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 4 });
    for (size_t i = 0; i < 4; ++i) t(i) = T(100 + i);
    t = T(0);
    for (size_t i = 0; i < 4; ++i) EXPECT_EQ(t(i), T(0));
  }

// operator== / operator!=

  TYPED_TEST(TensorTppTyped, EqualityShapeMismatch)
  {
    using T = TypeParam;
    sb::Tensor<T> a({ 2, 3 });
    sb::Tensor<T> b({ 3, 2 });
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
    sb::Tensor<T> dst({ 2, 3 });
    dst.assign_from(src);
    EXPECT_EQ(dst, src);
  }

  TYPED_TEST(TensorTppTyped, AssignFromShapeMismatchThrows)
  {
    using T = TypeParam;
    sb::Tensor<T> src({ 2, 3 });
    sb::Tensor<T> dst({ 3, 2 });
    EXPECT_THROW(dst.assign_from(src), std::invalid_argument);
  }

// size()

  TYPED_TEST(TensorTppTyped, SizeOfEmptyTensor)
  {
    using T = TypeParam;
    sb::Tensor<T> t;
    EXPECT_EQ(t.size(), 0u);
  }

  TYPED_TEST(TensorTppTyped, Size1d)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 7 });
    EXPECT_EQ(t.size(), 7u);
  }

  TYPED_TEST(TensorTppTyped, Size3d)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 2, 3, 5 });
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
    auto sliced = t[std::vector<sb::Slice>{ sb::range(1, 3, std::nullopt), sb::all() }];
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
    auto sliced = t[std::vector<sb::Slice>{ sb::range(0, 4, 2), sb::all() }];
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
    sb::walk(t, [&count](const std::vector<size_t>&) -> bool { return ++count < 5; });
    EXPECT_EQ(count, 5);
  }

  TYPED_TEST(TensorTppTyped, WalkBoolFunctorAllTrue)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4 });
    int count = 0;
    sb::walk(t, [&count](const std::vector<size_t>&) -> bool { ++count; return true; });
    EXPECT_EQ(count, 4);
  }

  TYPED_TEST(TensorTppTyped, WalkRank0ShapeVisitsSingleElement)
  {
    using T = TypeParam;
    // A default-constructed (rank-0, shape ()) tensor holds exactly one
    // element, so walk must visit it once with an empty index. Only a
    // zero-size extent is skipped (see WalkZeroDimensionDoesNothing).
    sb::Tensor<T> t;
    int count = 0;
    sb::walk(t, [&count](const std::vector<size_t>& idx) { ++count; EXPECT_TRUE(idx.empty()); });
    EXPECT_EQ(count, 1);
  }

  TYPED_TEST(TensorTppTyped, WalkZeroDimensionDoesNothing)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 0, 3 });
    int count = 0;
    sb::walk(t, [&count](const std::vector<size_t>&) { ++count; });
    EXPECT_EQ(count, 0);
  }

// apply()

  TYPED_TEST(TensorTppTyped, ApplyMultipliesAllElements)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 3, 3 });
    t.apply([](T& v) { v = v * T(2); });
    size_t n = 0;
    sb::walk(t, [&t, &n](const std::vector<size_t>& idx) {
      EXPECT_EQ(t(idx), T(n) * T(2));
      ++n;
    });
  }

  TYPED_TEST(TensorTppTyped, ApplyOnEmptyDimensionDoesNothing)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 0 });
    int calls = 0;
    t.apply([&calls](T&) { ++calls; });
    EXPECT_EQ(calls, 0);
  }

// extract() / operator[]

  TYPED_TEST(TensorTppTyped, SliceAllPreservesShape)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 3, 4 });
    auto view = t[std::vector<sb::Slice>{ sb::all(), sb::all() }];
    EXPECT_EQ(view.shape(0), 3u);
    EXPECT_EQ(view.shape(1), 4u);
  }

  TYPED_TEST(TensorTppTyped, SliceIndexDropsDimension)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 3, 4 });
    auto view = t[std::vector<sb::Slice>{ sb::index(1), sb::all() }];
    EXPECT_EQ(view.shape().size(), 1u);
    EXPECT_EQ(view.shape(0), 4u);
    for (size_t j = 0; j < 4; ++j)
      EXPECT_EQ(view({ j }), T(4 + j));
  }

  TYPED_TEST(TensorTppTyped, SliceRangeStopClampedToShape)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });
    auto view = t[std::vector<sb::Slice>{ sb::range(1, 100, std::nullopt) }];
    EXPECT_EQ(view.shape(0), 4u);
  }

  TYPED_TEST(TensorTppTyped, SliceRangeNegativeStartResolvesAgainstSize)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });  // [0, 1, 2, 3, 4]

    // An in-range negative start is resolved against the dimension size
    // (NumPy slice.indices): -2 -> 3, so [3, 4].
    auto view = t[std::vector<sb::Slice>{ sb::range(-2, std::nullopt, std::nullopt) }];
    EXPECT_EQ(view.shape(0), 2u);
    EXPECT_EQ(view({ 0 }), T(3));
    EXPECT_EQ(view({ 1 }), T(4));

    // A start more negative than the size clamps to 0 (not an error).
    auto clamped = t[std::vector<sb::Slice>{ sb::range(-10, 3, std::nullopt) }];
    EXPECT_EQ(clamped.shape(0), 3u);
    EXPECT_EQ(clamped({ 0 }), T(0));
    EXPECT_EQ(clamped({ 1 }), T(1));
    EXPECT_EQ(clamped({ 2 }), T(2));
  }

  TYPED_TEST(TensorTppTyped, SliceIndexNegativeResolvesAgainstSize)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4, 5 });
    // Row -1 is the last row.
    auto view = t[std::vector<sb::Slice>{ sb::index(-1), sb::all() }];
    EXPECT_EQ(view.shape().size(), 1u);
    EXPECT_EQ(view.shape(0), 5u);
    for (size_t j = 0; j < 5; ++j)
      EXPECT_EQ(view({ j }), T(3 * 5 + j));
  }

  TYPED_TEST(TensorTppTyped, SliceIndexOutOfBoundsThrows)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4, 5 });
    std::vector<sb::Slice> too_large{ sb::index(4), sb::all() };
    std::vector<sb::Slice> too_negative{ sb::index(-5), sb::all() };
    EXPECT_THROW((void)t[too_large], std::out_of_range);
    EXPECT_THROW((void)t[too_negative], std::out_of_range);
  }

  TYPED_TEST(TensorTppTyped, SliceRangeStopLessThanStartGivesZeroSize)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });
    auto view = t[std::vector<sb::Slice>{ sb::range(3, 1, std::nullopt) }];
    EXPECT_EQ(view.shape(0), 0u);
  }

  TYPED_TEST(TensorTppTyped, SliceRangeZeroStepThrows)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });
    // A zero step is invalid (NumPy: "slice step cannot be zero").
    EXPECT_THROW((void)t[std::vector<sb::Slice>{ sb::range(0, 5, 0) }],
                 std::invalid_argument);
  }

  TYPED_TEST(TensorTppTyped, SliceRangeNegativeStep)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });  // [0, 1, 2, 3, 4]
    auto view = t[std::vector<sb::Slice>{ sb::range(4, 0, -1) }];
    EXPECT_EQ(view.shape(0), 4u);
    EXPECT_EQ(view({0}), T(4));
    EXPECT_EQ(view({1}), T(3));
    EXPECT_EQ(view({2}), T(2));
    EXPECT_EQ(view({3}), T(1));
  }

  TYPED_TEST(TensorTppTyped, SliceRangeWithDefaultStartStop)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });
    auto view = t[std::vector<sb::Slice>{ sb::range(std::nullopt, std::nullopt, std::nullopt) }];
    EXPECT_EQ(view.shape(0), 5u);
    for (size_t i = 0; i < 5; ++i)
      EXPECT_EQ(view({ i }), T(i));
  }

  TYPED_TEST(TensorTppTyped, Slice3dMixed)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4, 5, 6 });
    auto view = t[std::vector<sb::Slice>{
        sb::index(1),
        sb::range(2, 4, std::nullopt),
        sb::range(std::nullopt, std::nullopt, 2)
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
    sb::Tensor<T> t({ 5 });
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
    EXPECT_EQ(sb::Tensor<T>({ 5 }).rank(), 1u);
    EXPECT_EQ(sb::Tensor<T>({ 3, 4 }).rank(), 2u);
    EXPECT_EQ(sb::Tensor<T>({ 2, 3, 5 }).rank(), 3u);
  }

  TYPED_TEST(TensorTppTyped, StridesCorrectForRowMajor2d)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3, 4 });
    ASSERT_EQ(t.strides().size(), 2u);
    EXPECT_EQ(t.stride(0), 4u);
    EXPECT_EQ(t.stride(1), 1u);
  }

  TYPED_TEST(TensorTppTyped, OffsetNonZeroAfterIndexSlice)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4, 4 });
    auto view = t[std::vector<sb::Slice>{ sb::index(1), sb::all() }];
    EXPECT_EQ(view.offset(), 4u);
  }

// ============================================================================
// Arithmetic operators
// ============================================================================

  TYPED_TEST(TensorTppTyped, DivideAssign)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(6); t(1) = T(4); t(2) = T(2);
    t /= T(2);
    EXPECT_EQ(t(0), T(3));
    EXPECT_EQ(t(1), T(2));
    EXPECT_EQ(t(2), T(1));
  }

  TYPED_TEST(TensorTppTyped, Divide)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(6); t(1) = T(4); t(2) = T(2);
    auto result = t / T(2);
    EXPECT_EQ(result(0), T(3));
    EXPECT_EQ(result(1), T(2));
    EXPECT_EQ(result(2), T(1));
    EXPECT_EQ(t(0), T(6));
    EXPECT_EQ(t(1), T(4));
    EXPECT_EQ(t(2), T(2));
  }

  TYPED_TEST(TensorTppTyped, MultiplyAssign)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(1); t(1) = T(2); t(2) = T(3);
    t *= T(4);
    EXPECT_EQ(t(0), T(4));
    EXPECT_EQ(t(1), T(8));
    EXPECT_EQ(t(2), T(12));
  }

  TYPED_TEST(TensorTppTyped, Multiply)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(1); t(1) = T(2); t(2) = T(3);
    auto result = t * T(4);
    EXPECT_EQ(result(0), T(4));
    EXPECT_EQ(result(1), T(8));
    EXPECT_EQ(result(2), T(12));
    EXPECT_EQ(t(0), T(1));
    EXPECT_EQ(t(1), T(2));
    EXPECT_EQ(t(2), T(3));
  }

  TYPED_TEST(TensorTppTyped, AddAssign)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(1); t(1) = T(2); t(2) = T(3);
    t += T(10);
    EXPECT_EQ(t(0), T(11));
    EXPECT_EQ(t(1), T(12));
    EXPECT_EQ(t(2), T(13));
  }

  TYPED_TEST(TensorTppTyped, Add)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(1); t(1) = T(2); t(2) = T(3);
    auto result = t + T(10);
    EXPECT_EQ(result(0), T(11));
    EXPECT_EQ(result(1), T(12));
    EXPECT_EQ(result(2), T(13));
    EXPECT_EQ(t(0), T(1));
    EXPECT_EQ(t(1), T(2));
    EXPECT_EQ(t(2), T(3));
  }

  TYPED_TEST(TensorTppTyped, SubtractAssign)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(5); t(1) = T(7); t(2) = T(9);
    t -= T(3);
    EXPECT_EQ(t(0), T(2));
    EXPECT_EQ(t(1), T(4));
    EXPECT_EQ(t(2), T(6));
  }

  TYPED_TEST(TensorTppTyped, Subtract)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(5); t(1) = T(7); t(2) = T(9);
    auto result = t - T(3);
    EXPECT_EQ(result(0), T(2));
    EXPECT_EQ(result(1), T(4));
    EXPECT_EQ(result(2), T(6));
    EXPECT_EQ(t(0), T(5));
    EXPECT_EQ(t(1), T(7));
    EXPECT_EQ(t(2), T(9));
  }

  TYPED_TEST(TensorTppTyped, FreeMultiply)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(1); t(1) = T(2); t(2) = T(3);
    auto result = T(4) * t;
    EXPECT_EQ(result(0), T(4));
    EXPECT_EQ(result(1), T(8));
    EXPECT_EQ(result(2), T(12));
    EXPECT_EQ(t(0), T(1));
    EXPECT_EQ(t(1), T(2));
    EXPECT_EQ(t(2), T(3));
  }

  TYPED_TEST(TensorTppTyped, FreeAdd)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(1); t(1) = T(2); t(2) = T(3);
    auto result = T(10) + t;
    EXPECT_EQ(result(0), T(11));
    EXPECT_EQ(result(1), T(12));
    EXPECT_EQ(result(2), T(13));
    EXPECT_EQ(t(0), T(1));
    EXPECT_EQ(t(1), T(2));
    EXPECT_EQ(t(2), T(3));
  }

  TYPED_TEST(TensorTppTyped, FreeSubtract)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(1); t(1) = T(2); t(2) = T(3);
    auto result = T(10) - t;
    EXPECT_EQ(result(0), T(9));
    EXPECT_EQ(result(1), T(8));
    EXPECT_EQ(result(2), T(7));
    EXPECT_EQ(t(0), T(1));
    EXPECT_EQ(t(1), T(2));
    EXPECT_EQ(t(2), T(3));
  }

  TYPED_TEST(TensorTppTyped, FreeDivide)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(1); t(1) = T(2); t(2) = T(4);
    auto result = T(8) / t;
    EXPECT_EQ(result(0), T(8));
    EXPECT_EQ(result(1), T(4));
    EXPECT_EQ(result(2), T(2));
    EXPECT_EQ(t(0), T(1));
    EXPECT_EQ(t(1), T(2));
    EXPECT_EQ(t(2), T(4));
  }

// ============================================================================
// Cross-type tests (not parameterizable on a single T)
// ============================================================================

  TEST(TensorTpp, AssignFromCrossTypeIntToDouble)
  {
    sb::Tensor<int> src({ 2, 2 });
    src({ 0, 0 }) = 1; src({ 0, 1 }) = 2;
    src({ 1, 0 }) = 3; src({ 1, 1 }) = 4;

    sb::Tensor<double> dst({ 2, 2 });
    dst.assign_from(src);

    EXPECT_DOUBLE_EQ(dst({ 0, 0 }), 1.0);
    EXPECT_DOUBLE_EQ(dst({ 0, 1 }), 2.0);
    EXPECT_DOUBLE_EQ(dst({ 1, 0 }), 3.0);
    EXPECT_DOUBLE_EQ(dst({ 1, 1 }), 4.0);
  }

  TEST(TensorTpp, AssignFromCrossTypeShapeMismatchThrows)
  {
    sb::Tensor<float> src({ 2, 3 });
    sb::Tensor<double> dst({ 3, 2 });
    EXPECT_THROW(dst.assign_from(src), std::invalid_argument);
  }

  TEST(TensorTpp, CrossTypeEqualityIntAndDouble)
  {
    sb::Tensor<int>    a({ 3 });
    sb::Tensor<double> b({ 3 });
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
    sb::Tensor<int>    a({ 3 });
    sb::Tensor<double> b({ 3 });
    for (size_t i = 0; i < 3; ++i)
    {
      a(i) = static_cast<int>(i);
      b(i) = static_cast<double>(i) + 0.5;
    }
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
  }

// ============================================================================
// allclose — Tensor
// ============================================================================

  using FloatTypes = ::testing::Types<float, double>;

  template <typename T>
  class AllcloseTensorTyped : public ::testing::Test {};
  TYPED_TEST_SUITE(AllcloseTensorTyped, FloatTypes);

  TYPED_TEST(AllcloseTensorTyped, IdenticalAreClose)
  {
    using T = TypeParam;
    auto a = make_sequential<T>({3, 4});
    EXPECT_TRUE(sb::allclose(a, a));
  }

  TYPED_TEST(AllcloseTensorTyped, ShapeMismatch)
  {
    using T = TypeParam;
    sb::Tensor<T> a({2, 3}, T(1));
    sb::Tensor<T> b({3, 2}, T(1));
    EXPECT_FALSE(sb::allclose(a, b));
  }

  TYPED_TEST(AllcloseTensorTyped, SmallPerturbationWithinTolerance)
  {
    using T = TypeParam;
    auto a = make_sequential<T>({3, 4});
    auto b = make_sequential<T>({3, 4});
    b({1, 2}) = b({1, 2}) + T(1e-9);
    EXPECT_TRUE(sb::allclose(a, b));
  }

  TYPED_TEST(AllcloseTensorTyped, LargePerturbationOutsideTolerance)
  {
    using T = TypeParam;
    auto a = make_sequential<T>({3, 4});
    auto b = make_sequential<T>({3, 4});
    b({2, 0}) = T(999);
    EXPECT_FALSE(sb::allclose(a, b));
  }

  TYPED_TEST(AllcloseTensorTyped, CustomAtol)
  {
    using T = TypeParam;
    sb::Tensor<T> a({2, 3}, T(0));
    sb::Tensor<T> b({2, 3}, T(0));
    b({1, 1}) = T(0.5);
    EXPECT_FALSE(sb::allclose(a, b));
    EXPECT_TRUE(sb::allclose(a, b, T(1)));
  }

  TYPED_TEST(AllcloseTensorTyped, EmptyTensorsAreClose)
  {
    using T = TypeParam;
    sb::Tensor<T> a({0});
    sb::Tensor<T> b({0});
    EXPECT_TRUE(sb::allclose(a, b));
  }

  TYPED_TEST(AllcloseTensorTyped, MemberDelegatesToFree)
  {
    using T = TypeParam;
    auto a = make_sequential<T>({2, 3});
    auto b = make_sequential<T>({2, 3});
    b({0, 1}) = b({0, 1}) + T(1e-9);
    EXPECT_EQ(a.allclose(b), sb::allclose(a, b));
  }

// ============================================================================
// allclose — compressed matrices (DistanceMatrix, SymmetricMatrix)
// ============================================================================

  template <typename T>
  struct CompressedMatrixTraits;

  template <typename T>
  struct CompressedMatrixTraits<sb::DistanceMatrix<T>>
  {
    using Scalar = T;
    static sb::DistanceMatrix<T> make(size_t n, T init) { return sb::DistanceMatrix<T>(n, init); }
    static void perturb(sb::DistanceMatrix<T>& m, T val) { m(0, 1) = val; }
  };

  template <typename T>
  struct CompressedMatrixTraits<sb::SymmetricMatrix<T>>
  {
    using Scalar = T;
    static sb::SymmetricMatrix<T> make(size_t n, T init) { return sb::SymmetricMatrix<T>(n, init); }
    static void perturb(sb::SymmetricMatrix<T>& m, T val) { m(0, 1) = val; }
  };

  using CompressedMatrixTypes = ::testing::Types<
    sb::DistanceMatrix<float>, sb::DistanceMatrix<double>,
    sb::SymmetricMatrix<float>, sb::SymmetricMatrix<double>>;

  template <typename T>
  class AllcloseCompressedTyped : public ::testing::Test {};
  TYPED_TEST_SUITE(AllcloseCompressedTyped, CompressedMatrixTypes);

  TYPED_TEST(AllcloseCompressedTyped, IdenticalAreClose)
  {
    using Traits = CompressedMatrixTraits<TypeParam>;
    using T = typename Traits::Scalar;
    auto a = Traits::make(3, T(1));
    auto b = Traits::make(3, T(1));
    EXPECT_TRUE(sb::allclose(a, b));
  }

  TYPED_TEST(AllcloseCompressedTyped, SizeMismatch)
  {
    using Traits = CompressedMatrixTraits<TypeParam>;
    using T = typename Traits::Scalar;
    auto a = Traits::make(3, T(1));
    auto b = Traits::make(4, T(1));
    EXPECT_FALSE(sb::allclose(a, b));
  }

  TYPED_TEST(AllcloseCompressedTyped, SmallPerturbationWithinTolerance)
  {
    using Traits = CompressedMatrixTraits<TypeParam>;
    using T = typename Traits::Scalar;
    auto a = Traits::make(3, T(1));
    auto b = Traits::make(3, T(1));
    Traits::perturb(b, T(1) + T(1e-9));
    EXPECT_TRUE(sb::allclose(a, b));
  }

  TYPED_TEST(AllcloseCompressedTyped, LargePerturbationOutsideTolerance)
  {
    using Traits = CompressedMatrixTraits<TypeParam>;
    using T = typename Traits::Scalar;
    auto a = Traits::make(3, T(1));
    auto b = Traits::make(3, T(1));
    Traits::perturb(b, T(5));
    EXPECT_FALSE(sb::allclose(a, b));
  }

  TYPED_TEST(AllcloseCompressedTyped, CustomAtol)
  {
    using Traits = CompressedMatrixTraits<TypeParam>;
    using T = typename Traits::Scalar;
    auto a = Traits::make(3, T(0));
    auto b = Traits::make(3, T(0));
    Traits::perturb(b, T(0.5));
    EXPECT_FALSE(sb::allclose(a, b));
    EXPECT_TRUE(sb::allclose(a, b, T(1)));
  }

  TEST(AllcloseCompressed, SymmetricDiagonalDifferenceIsCompared)
  {
    // The elementwise loop must include the diagonal: it is real stored data
    // for a symmetric matrix (a distance matrix reads zero there either way).
    sb::SymmetricMatrix<double> a(3, 1.0);
    sb::SymmetricMatrix<double> b(3, 1.0);
    b(1, 1) = 2.0;
    EXPECT_FALSE(sb::allclose(a, b));
    EXPECT_TRUE(sb::allclose(a, b, 1.0));
  }

  // ==========================================================================
  // DistanceMatrix indexed views — buffer access
  // ==========================================================================

  TEST(DistanceMatrixView, DataThrowsOnIndexedViewSourceDataDoesNot)
  {
    // data() on a view is the footgun that once corrupted serialization: the
    // buffer is the full shared source while size()/storage_count() describe
    // the submatrix. It must throw; deliberate source access goes through
    // source_data() (paired with source_size()).
    sb::DistanceMatrix<double> source(4);
    source(0, 1) = 1.0;
    source(1, 3) = 2.0;

    sb::Tensor<uint64_t> indices({2});
    indices({0}) = 1;
    indices({1}) = 3;
    sb::DistanceMatrix<double> view(source, indices);

    EXPECT_THROW((void)view.data(), std::logic_error);
    EXPECT_EQ(view.source_data(), source.data());
    EXPECT_EQ(view.source_size(), 4u);
  }

  TEST(DistanceMatrixView, StorageCountReportsTheObservedSubmatrix)
  {
    // storage_count() describes the matrix as observed, like size() and
    // operator(): for a view, the entry count of its principal submatrix.
    // The physical source count stays available as storage_size(source_size()).
    sb::DistanceMatrix<double> source(4);
    EXPECT_EQ(source.storage_count(), 6u);

    sb::Tensor<uint64_t> indices({2});
    indices({0}) = 1;
    indices({1}) = 3;
    sb::DistanceMatrix<double> view(source, indices);
    EXPECT_EQ(view.storage_count(), 1u);
    EXPECT_EQ(sb::DistanceMatrix<double>::storage_size(view.source_size()), 6u);
  }

  TEST(DistanceMatrixView, AllcloseComparesThePrincipalSubmatrix)
  {
    // allclose on a view must compare the selected submatrix, not the raw
    // shared source buffer (which data() no longer even hands out).
    sb::DistanceMatrix<double> source(4);
    source(0, 1) = 1.0;
    source(0, 2) = 2.0;
    source(0, 3) = 3.0;
    source(1, 2) = 4.0;
    source(1, 3) = 5.0;
    source(2, 3) = 6.0;

    sb::Tensor<uint64_t> indices({2});
    indices({0}) = 1;
    indices({1}) = 3;
    sb::DistanceMatrix<double> view(source, indices);  // submatrix {{0, 5}, {5, 0}}

    EXPECT_TRUE(sb::allclose(view, view.materialize()));

    sb::DistanceMatrix<double> other(2);
    other(0, 1) = 5.0;
    EXPECT_TRUE(sb::allclose(view, other));
    other(0, 1) = 5.5;
    EXPECT_FALSE(sb::allclose(view, other));
    EXPECT_TRUE(sb::allclose(view, other, 1.0));

    sb::DistanceMatrix<double> wrongSize(3);
    EXPECT_FALSE(sb::allclose(view, wrongSize));
  }

  TEST(DistanceMatrixView, CopyKeepsOrDetachesTheSource)
  {
    sb::DistanceMatrix<double> source(4);
    source(1, 3) = 5.0;

    sb::Tensor<uint64_t> indices({2});
    indices({0}) = 1;
    indices({1}) = 3;
    sb::DistanceMatrix<double> view(source, indices);

    // const: a non-const read through operator() would COW-detach the copies.
    const auto shared = view.copy();        // default: keeps sharing the source
    const auto detached = view.copy(false); // clean slate: source deep-copied too
    EXPECT_TRUE(shared.is_indexed());
    EXPECT_TRUE(detached.is_indexed());
    EXPECT_EQ(shared.source_data(), source.data());
    EXPECT_NE(detached.source_data(), source.data());
    EXPECT_EQ(detached(0, 1), 5.0);

    // Moving the source shows through the sharing copy but not the detached one.
    source(1, 3) = 7.0;
    EXPECT_EQ(shared(0, 1), 7.0);
    EXPECT_EQ(detached(0, 1), 5.0);
  }

  TEST(DistanceMatrixView, WriteDetachesTheViewFromTheSource)
  {
    // Copy-on-write: writing through an indexed view materializes it in place
    // first, so the write lands on a private buffer and the shared source
    // stays untouched.
    sb::DistanceMatrix<double> source(4);
    source(1, 3) = 5.0;

    sb::Tensor<uint64_t> indices({2});
    indices({0}) = 1;
    indices({1}) = 3;
    sb::DistanceMatrix<double> view(source, indices);
    EXPECT_TRUE(view.is_indexed());

    view(0, 1) = 9.0;
    EXPECT_FALSE(view.is_indexed());
    EXPECT_EQ(view(0, 1), 9.0);
    EXPECT_EQ(source(1, 3), 5.0);
  }

  TEST(PointCloudView, CopyKeepsOrDetachesTheSource)
  {
    sb::Tensor<double> coords({3, 2});
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 2; ++j)
        coords({i, j}) = static_cast<double>((2 * i) + j);

    sb::Tensor<uint64_t> indices({2});
    indices({0}) = 1;
    indices({1}) = 2;
    sb::PointCloud<double> view(coords, indices);

    auto shared = view.copy();
    auto detached = view.copy(false);
    EXPECT_TRUE(shared.is_indexed());
    EXPECT_TRUE(detached.is_indexed());
    EXPECT_EQ(shared.coords().data(), coords.data());
    EXPECT_NE(detached.coords().data(), coords.data());
    EXPECT_EQ(detached(0, 0), 2.0);  // point 1 = (2, 3)

    coords({1, 0}) = 9.0;
    EXPECT_EQ(shared(0, 0), 9.0);
    EXPECT_EQ(detached(0, 0), 2.0);
  }

  TEST(PointCloudView, WalkVisitsIndexedCellsTransparently)
  {
    // walk only drives the shape odometer, so a tensor whose cells are indexed
    // views must walk exactly like one with owning cells: every cell visited
    // once, in order, with reads through the cell giving the selected points.
    sb::Tensor<double> coords({4, 2});
    for (size_t i = 0; i < 4; ++i)
      for (size_t j = 0; j < 2; ++j)
        coords({i, j}) = static_cast<double>((10 * i) + j);

    sb::Tensor<uint64_t> idx0({2});
    idx0({0}) = 3;
    idx0({1}) = 1;
    sb::Tensor<uint64_t> idx1({1});
    idx1({0}) = 2;

    sb::Tensor<sb::PointCloud<double>> t({3});
    t(0) = sb::PointCloud<double>(coords, idx0);
    t(1) = sb::PointCloud<double>(coords, idx1);
    t(2) = sb::PointCloud<double>(coords); // owning cell over the same buffer

    std::vector<std::vector<size_t>> visited;
    std::vector<size_t> nPoints;
    std::vector<double> firstCoord;
    sb::walk(t, [&](const std::vector<size_t>& idx) {
      visited.push_back(idx);
      nPoints.push_back(t(idx).n_points());
      firstCoord.push_back(t(idx)(0, 0));
    });

    std::vector<std::vector<size_t>> expectedOrder = {{0}, {1}, {2}};
    EXPECT_EQ(visited, expectedOrder);
    std::vector<size_t> expectedNPoints = {2, 1, 4};
    EXPECT_EQ(nPoints, expectedNPoints);
    std::vector<double> expectedFirst = {30.0, 20.0, 0.0}; // first selected point of each cell
    EXPECT_EQ(firstCoord, expectedFirst);
  }

  TEST(PointCloudView, PcloudCastPreservesSourceSharing)
  {
    // pcloud_cast must not materialize views: the source buffer is cast once
    // and the views are rebuilt on top of it, so a cast subsample tensor keeps
    // the one-source-plus-index-arrays storage layout.
    sb::Tensor<double> coords({3, 2});
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 2; ++j)
        coords({i, j}) = static_cast<double>((2 * i) + j) + 0.5;

    sb::Tensor<uint64_t> idx0({2});
    idx0({0}) = 1;
    idx0({1}) = 2;
    sb::Tensor<uint64_t> idx1({3});
    idx1({0}) = 0;
    idx1({1}) = 0;
    idx1({2}) = 2;

    sb::Tensor<sb::PointCloud<double>> t({3});
    t(0) = sb::PointCloud<double>(coords, idx0);
    t(1) = sb::PointCloud<double>(coords, idx1);
    t(2) = sb::PointCloud<double>(coords); // owning cell over the same buffer

    auto cast = sb::pcloud_cast<float>(t);

    // Same points as casting each cell's materialization (the old behavior).
    for (size_t k = 0; k < 3; ++k)
    {
      sb::PointCloud<float> expected(sb::tensor_cast<float>(t(k).materialize()));
      EXPECT_TRUE(cast(k) == expected) << "cell " << k;
    }

    // One shared cast buffer, view flags intact.
    EXPECT_TRUE(cast(0).is_indexed());
    EXPECT_TRUE(cast(1).is_indexed());
    EXPECT_FALSE(cast(2).is_indexed());
    EXPECT_EQ(cast(0).coords().data(), cast(1).coords().data());
    EXPECT_EQ(cast(0).coords().data(), cast(2).coords().data());

    // Index arrays carry over by value, not by aliasing the source tensor's.
    EXPECT_TRUE(cast(0).indices() == t(0).indices());
    EXPECT_NE(cast(0).indices().data(), t(0).indices().data());
  }

  TEST(PointCloudView, PcloudCastKeepsDistinctSourcesDistinct)
  {
    // Dedup is per source buffer: cells over different sources must not be
    // collapsed onto one cast buffer.
    sb::Tensor<double> coordsA({2, 1});
    coordsA({0, 0}) = 1.0;
    coordsA({1, 0}) = 2.0;
    sb::Tensor<double> coordsB({2, 1});
    coordsB({0, 0}) = 3.0;
    coordsB({1, 0}) = 4.0;

    sb::Tensor<uint64_t> indices({1});
    indices({0}) = 1;

    sb::Tensor<sb::PointCloud<double>> t({2});
    t(0) = sb::PointCloud<double>(coordsA, indices);
    t(1) = sb::PointCloud<double>(coordsB, indices);

    auto cast = sb::pcloud_cast<float>(t);
    EXPECT_NE(cast(0).coords().data(), cast(1).coords().data());
    EXPECT_EQ(cast(0)(0, 0), 2.0f);
    EXPECT_EQ(cast(1)(0, 0), 4.0f);
  }

} // namespace
