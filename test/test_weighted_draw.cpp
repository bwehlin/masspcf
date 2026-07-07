#include <gtest/gtest.h>
#include <sbear/executor.hpp>
#include <sbear/sampling/weighted_draw.hpp>
#include <sbear/sampling/weighting.hpp>
#include <sbear/xoroshiro128pp.hpp>

#include <cstdint>
#include <limits>
#include <set>
#include <span>
#include <vector>

// Unit tests for the row-level weighted-draw primitives underlying
// sample_subsets: one row is prepared per query point and then drawn from once
// per subsample. End-to-end behavior (determinism, generator advancement,
// region semantics) is covered by the Python tests in test/python/sampling/;
// these tests cover the primitives' contracts in isolation.

namespace
{
  using sb::sampling::detail::draw_indices;
  using sb::sampling::detail::draw_with_replacement;
  using sb::sampling::detail::draw_without_replacement;
  using sb::sampling::detail::index_for_target;
  using sb::sampling::detail::prepare_weight_matrix;
  using sb::sampling::detail::prepare_weight_row;
  using sb::sampling::detail::weight_row;

  sb::Xoroshiro128pp engine(uint64_t seed = 7)
  {
    return sb::Xoroshiro128pp(seed, seed + 1);
  }

  std::vector<uint64_t> to_vector(const sb::Tensor<uint64_t>& drawn)
  {
    std::vector<uint64_t> indices(drawn.size());
    for (size_t i = 0; i < drawn.size(); ++i)
      indices[i] = drawn({i});
    return indices;
  }

  std::vector<double> to_vector_row(const sb::Tensor<double>& matrix, size_t row)
  {
    std::vector<double> values(matrix.shape(1));
    for (size_t j = 0; j < values.size(); ++j)
      values[j] = matrix({row, j});
    return values;
  }
}

// ---------------------------------------------------------------------------
// prepare_weight_row
// ---------------------------------------------------------------------------

TEST(PrepareWeightRow, CountsStrictlyPositiveWeightsAsEligible)
{
  std::vector<double> row{0.5, 0.0, 2.0, 0.0, 1.0};
  EXPECT_EQ(prepare_weight_row(std::span<double>(row), false), 3u);
  // Without CDF conversion the weights are left untouched.
  EXPECT_EQ(row, (std::vector<double>{0.5, 0.0, 2.0, 0.0, 1.0}));
}

TEST(PrepareWeightRow, ConvertsRowToPrefixSumsWhenRequested)
{
  std::vector<double> row{0.5, 0.0, 2.0, 0.0, 1.0};
  EXPECT_EQ(prepare_weight_row(std::span<double>(row), true), 3u);
  EXPECT_EQ(row, (std::vector<double>{0.5, 0.5, 2.5, 2.5, 3.5}));
}

TEST(PrepareWeightRow, AllZeroRowIsAValidEmptyRegion)
{
  std::vector<double> row{0.0, 0.0, 0.0};
  EXPECT_EQ(prepare_weight_row(std::span<double>(row), true), 0u);
}

TEST(PrepareWeightRow, RejectsNegativeWeights)
{
  std::vector<double> row{1.0, -0.5, 2.0};
  EXPECT_THROW(prepare_weight_row(std::span<double>(row), false), std::invalid_argument);
}

TEST(PrepareWeightRow, RejectsNanWeights)
{
  std::vector<double> row{1.0, std::numeric_limits<double>::quiet_NaN()};
  EXPECT_THROW(prepare_weight_row(std::span<double>(row), true), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// index_for_target
// ---------------------------------------------------------------------------

TEST(IndexForTarget, BoundaryTargetSkipsTrailingZeroWeights)
{
  // uniform_real_distribution may return exactly cdf.back() (LWG 2524). Such
  // a draw must land on the last positive-weight interval — clamping to the
  // last *array index* instead would put an ineligible trailing-zero point
  // into the subsample (or a duplicate, once no-replacement draws zero out
  // the tail of the row).
  std::vector<double> row{2.0, 3.0, 0.0, 0.0};
  prepare_weight_row(std::span<double>(row), true);  // cdf = {2, 5, 5, 5}

  EXPECT_EQ(index_for_target(std::span<const double>(row), 5.0), 1u);
}

TEST(IndexForTarget, InteriorTargetsMapToTheirIntervals)
{
  // Zero-width intervals (leading, interior) are skipped by upper_bound; a
  // target on an interior boundary belongs to the next interval.
  std::vector<double> row{0.0, 2.0, 0.0, 3.0};
  prepare_weight_row(std::span<double>(row), true);  // cdf = {0, 2, 2, 5}

  EXPECT_EQ(index_for_target(std::span<const double>(row), 0.0), 1u);
  EXPECT_EQ(index_for_target(std::span<const double>(row), 1.5), 1u);
  EXPECT_EQ(index_for_target(std::span<const double>(row), 2.0), 3u);
  EXPECT_EQ(index_for_target(std::span<const double>(row), 4.9), 3u);
}

// ---------------------------------------------------------------------------
// draw_with_replacement
// ---------------------------------------------------------------------------

TEST(DrawWithReplacement, DrawsExactlySampleSizeInRange)
{
  std::vector<double> row{1.0, 1.0, 0.0, 1.0};
  prepare_weight_row(std::span<double>(row), true);

  auto eng = engine();
  auto drawn = to_vector(draw_with_replacement(std::span<const double>(row), 32, eng));

  ASSERT_EQ(drawn.size(), 32u);
  for (uint64_t idx : drawn)
    EXPECT_LT(idx, 4u);
}

TEST(DrawWithReplacement, NeverDrawsZeroWeightPoints)
{
  // Zero weights produce zero-width CDF intervals, which upper_bound skips.
  std::vector<double> row{0.0, 1.0, 0.0, 1.0, 0.0};
  prepare_weight_row(std::span<double>(row), true);

  auto eng = engine();
  for (uint64_t idx : to_vector(draw_with_replacement(std::span<const double>(row), 200, eng)))
    EXPECT_TRUE(idx == 1u || idx == 3u) << "drew ineligible index " << idx;
}

// ---------------------------------------------------------------------------
// draw_without_replacement
// ---------------------------------------------------------------------------

TEST(DrawWithoutReplacement, DrawsAreDistinctAndEligible)
{
  std::vector<double> row{1.0, 0.0, 2.0, 3.0, 0.0, 1.0};

  auto eng = engine();
  auto drawn = to_vector(draw_without_replacement(std::span<const double>(row), 4, eng));

  ASSERT_EQ(drawn.size(), 4u);
  EXPECT_EQ(std::set<uint64_t>(drawn.begin(), drawn.end()).size(), 4u);
  for (uint64_t idx : drawn)
    EXPECT_GT(row[idx], 0.0) << "drew ineligible index " << idx;
}

TEST(DrawWithoutReplacement, DrawingAllEligiblePointsYieldsEachOnce)
{
  std::vector<double> row{0.5, 0.0, 1.5, 2.5};

  auto eng = engine();
  auto drawn = to_vector(draw_without_replacement(std::span<const double>(row), 3, eng));

  EXPECT_EQ(std::set<uint64_t>(drawn.begin(), drawn.end()),
            (std::set<uint64_t>{0u, 2u, 3u}));
}

// ---------------------------------------------------------------------------
// draw_indices (the ragged-length dispatch over the modes)
// ---------------------------------------------------------------------------

TEST(DrawIndices, EmptyRegionGivesLengthZeroSubsample)
{
  std::vector<double> row{0.0, 0.0};
  auto eng = engine();
  EXPECT_EQ(draw_indices(std::span<const double>(row), 0, 5, true, eng).size(), 0u);
}

TEST(DrawIndices, WithReplacementFillsSampleSize)
{
  std::vector<double> row{1.0, 1.0};
  const size_t nEligible = prepare_weight_row(std::span<double>(row), true);

  auto eng = engine();
  EXPECT_EQ(draw_indices(std::span<const double>(row), nEligible, 5, true, eng).size(), 5u);
}

TEST(DrawIndices, WithoutReplacementShrinksToEligibleCount)
{
  std::vector<double> row{1.0, 0.0, 1.0};
  const size_t nEligible = prepare_weight_row(std::span<double>(row), false);

  auto eng = engine();
  EXPECT_EQ(draw_indices(std::span<const double>(row), nEligible, 5, false, eng).size(), 2u);
}

// ---------------------------------------------------------------------------
// prepare_weight_matrix (blocking parallel preparation of all rows)
// ---------------------------------------------------------------------------

TEST(PrepareWeightMatrix, PreparesEveryRowAndCountsEligibles)
{
  sb::Tensor<double> weights({2, 3});
  weights({0, 0}) = 1.0; weights({0, 1}) = 0.0; weights({0, 2}) = 2.0;
  weights({1, 0}) = 0.0; weights({1, 1}) = 0.0; weights({1, 2}) = 0.0;

  auto nEligible = prepare_weight_matrix(weights, /*toCdf=*/true, sb::default_executor());

  EXPECT_EQ(nEligible({0}), 2u);
  EXPECT_EQ(nEligible({1}), 0u);  // all-zero row: a valid empty region
  // Row 0 was converted in place to its CDF; the empty row is untouched.
  EXPECT_EQ(to_vector_row(weights, 0), (std::vector<double>{1.0, 1.0, 3.0}));
  EXPECT_EQ(to_vector_row(weights, 1), (std::vector<double>{0.0, 0.0, 0.0}));
}

TEST(PrepareWeightMatrix, ValidationErrorsOnWorkersReachTheCaller)
{
  sb::Tensor<double> weights({2, 2});
  weights({0, 0}) = 1.0; weights({0, 1}) = 1.0;
  weights({1, 0}) = 1.0; weights({1, 1}) = -1.0;

  EXPECT_THROW(prepare_weight_matrix(weights, true, sb::default_executor()),
               std::invalid_argument);
}
