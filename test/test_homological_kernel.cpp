#include <gtest/gtest.h>

#include <sbear/distance_matrix.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/persistence/compute_homological_kernel.hpp>
#include <sbear/persistence/persistence_pair.hpp>
#include <sbear/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
  using ScalarTypes = ::testing::Types<sb::float32_t, sb::float64_t>;

  template <typename T>
  class HomologicalKernelTest : public ::testing::Test
  {
  };

  TYPED_TEST_SUITE(HomologicalKernelTest, ScalarTypes);

  template <typename T>
  constexpr T tolerance()
  {
    return std::is_same_v<T, sb::float32_t> ? static_cast<T>(1e-5) : static_cast<T>(1e-12);
  }

  // ============================================================================
  // Helpers
  // ============================================================================

  /// Build a DistanceMatrix<T> from a full square matrix given row by row.
  template <typename T>
  sb::DistanceMatrix<T> make_distmat(const std::vector<std::vector<T>> &full)
  {
    const auto n = full.size();
    sb::DistanceMatrix<T> dm(n);
    for (size_t i = 0; i < n; ++i)
    {
      EXPECT_EQ(full[i].size(), n) << "make_distmat input must be square";
      for (size_t j = i + 1; j < n; ++j)
      {
        EXPECT_EQ(full[i][j], full[j][i]) << "make_distmat input must be symmetric";
        dm(i, j) = full[i][j];
      }
    }
    return dm;
  }

  /// Build a PointCloud<T> (rank-2 tensor, one point per row) from a list of points.
  template <typename T>
  sb::PointCloud<T> make_pcloud(const std::vector<std::vector<T>> &points)
  {
    const auto n = points.size();
    const auto dim = points.empty() ? size_t{0} : points[0].size();
    sb::PointCloud<T> pc(std::vector<size_t>{n, dim});
    for (size_t i = 0; i < n; ++i)
    {
      for (size_t k = 0; k < dim; ++k)
      {
        pc({i, k}) = points[i][k];
      }
    }
    return pc;
  }

  /// Run the algorithm core on a (d, d') distance-matrix pair.
  template <typename T>
  sb::ph::Barcode<T> kernel_of(const sb::DistanceMatrix<T> &d, const sb::DistanceMatrix<T> &dPrime)
  {
    sb::ph::Barcode<T> bc;
    sb::ph::detail::homological_kernel_single_impl(d, dPrime, d.size(), bc);
    return bc;
  }

  /// Run the algorithm core on a (X, X') point-cloud pair via the Euclidean oracle.
  template <typename T>
  sb::ph::Barcode<T> kernel_of(const sb::PointCloud<T> &X, const sb::PointCloud<T> &XPrime)
  {
    sb::ph::detail::SquaredEuclideanDistance<T> d(X);
    sb::ph::detail::SquaredEuclideanDistance<T> dPrime(XPrime);
    sb::ph::Barcode<T> bc;
    sb::ph::detail::homological_kernel_single_impl(d, dPrime, d.size(), bc, [](T v) { return std::sqrt(v); });
    return bc;
  }

  /// Compare a computed barcode against expected {birth, death} bars via
  /// Barcode::is_isomorphic_to (order-independent, default tolerances).
  template <typename T>
  void expect_bars(const sb::ph::Barcode<T> &bc, const std::vector<std::pair<T, T>> &expected)
  {
    std::vector<sb::ph::PersistencePair<T>> bars;
    bars.reserve(expected.size());
    for (const auto &[birth, death] : expected)
    {
      bars.emplace_back(birth, death);
    }
    sb::ph::Barcode<T> expectedBc(std::move(bars));

    EXPECT_TRUE(bc.is_isomorphic_to(expectedBc)) << "actual:   " << bc << "\nexpected: " << expectedBc;
  }

  /// Check the kernel barcode of a (d, d') distance-matrix pair against expected bars.
  template <typename T>
  void expect_kernel_bars(
      const std::vector<std::vector<T>> &d, const std::vector<std::vector<T>> &dPrime,
      const std::vector<std::pair<T, T>> &expected)
  {
    expect_bars(kernel_of(make_distmat(d), make_distmat(dPrime)), expected);
  }

  // Shared hand example: four points on a line at 0, 1, 3, 7.
  // d merges: {0,1}@1, {1,2}@2, {2,3}@4. Cophenetic d_sd:
  //   d_sd(0,1)=1, d_sd(0,2)=d_sd(1,2)=2, d_sd(*,3)=4.
  template <typename T>
  std::vector<std::vector<T>> line_metric()
  {
    return {
        {T(0), T(1), T(3), T(7)},
        {T(1), T(0), T(2), T(6)},
        {T(3), T(2), T(0), T(4)},
        {T(7), T(6), T(4), T(0)},
    };
  }

  // ============================================================================
  // SquaredEuclideanDistance oracle
  // ============================================================================

  TYPED_TEST(HomologicalKernelTest, SquaredEuclideanDistanceMatchesHandComputed)
  {
    using T = TypeParam;

    auto pc = make_pcloud<T>({{T(0), T(0)}, {T(3), T(4)}, {T(0), T(12)}});
    sb::ph::detail::SquaredEuclideanDistance<T> dist(pc);

    EXPECT_EQ(dist.size(), 3U);
    EXPECT_NEAR(dist(0, 1), T(25), tolerance<T>());
    EXPECT_NEAR(dist(0, 2), T(144), tolerance<T>());
    EXPECT_NEAR(dist(1, 2), T(73), tolerance<T>());
    EXPECT_NEAR(dist(2, 1), dist(1, 2), tolerance<T>()); // symmetric
    EXPECT_EQ(dist(1, 1), T(0));
  }

  // ============================================================================
  // Core algorithm (homological_kernel_single_impl) on hand-calculated metrics
  // ============================================================================

  TYPED_TEST(HomologicalKernelTest, HalvedMetricGivesDoublingBars)
  {
    using T = TypeParam;

    // d' = d/2 keeps the merge structure, so every bar is [w, 2w) for the
    // d'-merge scales w = 0.5, 1, 2 of the line example.
    auto d = line_metric<T>();
    auto dPrime = d;
    for (auto &row : dPrime)
    {
      for (auto &v : row)
      {
        v /= T(2);
      }
    }

    expect_kernel_bars<T>(d, dPrime, {{T(0.5), T(1)}, {T(1), T(2)}, {T(2), T(4)}});
  }

  TYPED_TEST(HomologicalKernelTest, IdenticalMetricsGiveEmptyBars)
  {
    using T = TypeParam;

    // d' == d: every death equals its birth — empty bars [w, w), no throw.
    auto d = line_metric<T>();
    expect_kernel_bars<T>(d, d, {{T(1), T(1)}, {T(2), T(2)}, {T(4), T(4)}});
  }

  TYPED_TEST(HomologicalKernelTest, DeathCarriedByNonMergedPair)
  {
    using T = TypeParam;

    // Points x=0, y=1, a=2, b=3. d is the ultrametric of the tree
    // (((x,y)@1, a)@5, b)@9. The d' merges are (a,x)@0.1, (b,y)@0.2, then
    // {a,x} joins {b,y} at 0.5 via the edge (a,b). The death of that third bar
    // is min cross-pair d_sd = d_sd(x,y) = 1, carried by the pair (x,y) — which
    // involves neither endpoint of the merge edge: the death is determined by
    // the whole component pair.
    std::vector<std::vector<T>> d = {
        {T(0), T(1), T(5), T(9)},
        {T(1), T(0), T(5), T(9)},
        {T(5), T(5), T(0), T(9)},
        {T(9), T(9), T(9), T(0)},
    };
    std::vector<std::vector<T>> dPrime = {
        {T(0.0), T(0.9), T(0.1), T(0.9)},
        {T(0.9), T(0.0), T(0.9), T(0.2)},
        {T(0.1), T(0.9), T(0.0), T(0.5)},
        {T(0.9), T(0.2), T(0.5), T(0.0)},
    };

    expect_kernel_bars<T>(d, dPrime, {{T(0.1), T(5)}, {T(0.2), T(9)}, {T(0.5), T(1)}});
  }

  // Pure merge-order inversion (abstract metrics): d' merges A-B first, but
  // in d the tight pair is B-C. Deaths come from d, order from d'.
  TYPED_TEST(HomologicalKernelTest, MergeOrderInversion)
  {
    using T = TypeParam;

    std::vector<std::vector<T>> dPrime = {
        {T(0), T(1), T(3)},
        {T(1), T(0), T(2)},
        {T(3), T(2), T(0)},
    };
    std::vector<std::vector<T>> d = {
        {T(0), T(5), T(6)},
        {T(5), T(0), T(3)},
        {T(6), T(3), T(0)},
    };

    expect_kernel_bars<T>(d, dPrime, {{T(1), T(5)}, {T(2), T(3)}});
  }

  // Two disjoint pairs merge at the same birth (tie). The barcode multiset
  // must be independent of which tied merge is processed first.
  // Note d_sd(C,D) = 4 (MST path C-A-D), not the raw 5.
  TYPED_TEST(HomologicalKernelTest, TieInvariantBarcode)
  {
    using T = TypeParam;

    std::vector<std::vector<T>> dPrime = {
        {T(0), T(1), T(2), T(2)},
        {T(1), T(0), T(2), T(2)},
        {T(2), T(2), T(0), T(1)},
        {T(2), T(2), T(1), T(0)},
    };
    std::vector<std::vector<T>> d = {
        {T(0), T(3), T(4), T(4)},
        {T(3), T(0), T(4), T(4)},
        {T(4), T(4), T(0), T(5)},
        {T(4), T(4), T(5), T(0)},
    };

    expect_kernel_bars<T>(d, dPrime, {{T(1), T(3)}, {T(1), T(4)}, {T(2), T(4)}});
  }

  // Contracting a d-far pair creates a shortcut that lowers the death of a
  // LATER merge between two other components. Points in the projection frame
  // (x = the coordinate d' keeps): A=(0,6), P=(5,6), P'=(5,0), B=(2,0); d'
  // drops y, so P and P' contract at d'=0. At the merge (A,B)@2 the death is 5,
  // via the chain A -5- P ≡ P' -3- B — a path through a component containing
  // neither A nor B: deaths are computed against the quotient space.
  TYPED_TEST(HomologicalKernelTest, ContractionShortcutLowersLaterDeath)
  {
    using T = TypeParam;

    auto X = make_pcloud<T>({{T(0), T(6)}, {T(5), T(6)}, {T(5), T(0)}, {T(2), T(0)}});
    auto XPrime = make_pcloud<T>({{T(0), T(0)}, {T(5), T(0)}, {T(5), T(0)}, {T(2), T(0)}});

    expect_bars(kernel_of(X, XPrime), {{T(0), T(6)}, {T(2), T(5)}, {T(3), T(3)}});
  }

  // Staircase version of the shortcut above: the death chain of the (A,B)@1
  // merge passes through TWO earlier contractions,
  // A -8- R1 ≡ R1' -3- R2 ≡ R2' -4- B (a k-rung staircase needs k hops).
  // The deaths equal the d-MST weight multiset {3, 4, 8, 10, 10}.
  TYPED_TEST(HomologicalKernelTest, StaircaseDeathNeedsChainedContractionShortcuts)
  {
    using T = TypeParam;

    auto X = make_pcloud<T>({
        {T(0), T(20)}, // A
        {T(8), T(20)}, // R1
        {T(8), T(10)}, // R1'
        {T(5), T(10)}, // R2
        {T(5), T(0)},  // R2'
        {T(1), T(0)},  // B
    });
    auto XPrime = make_pcloud<T>({
        {T(0), T(0)},
        {T(8), T(0)},
        {T(8), T(0)},
        {T(5), T(0)},
        {T(5), T(0)},
        {T(1), T(0)},
    });

    expect_bars(kernel_of(X, XPrime), {{T(0), T(10)}, {T(0), T(10)}, {T(1), T(8)}, {T(3), T(3)}, {T(4), T(4)}});
  }

  // The births of ker mu are exactly the d'-MST edge weights and the deaths
  // are a permutation of the d-MST edge weights (mu_t is surjective, so the
  // number of bars alive at t is #components_d(t) - #components_d'(t) for
  // every t); only the birth-death pairing is nontrivial. This pins both
  // multisets exactly on random dominated inputs.
  TYPED_TEST(HomologicalKernelTest, RandomInputsPairMstEdgeWeightsExactly)
  {
    using T = TypeParam;

    std::mt19937 rng(20260716U);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    for (int trial = 0; trial < 50; ++trial)
    {
      const size_t n = 2 + static_cast<size_t>(rng() % 23);
      sb::DistanceMatrix<T> d(n);
      sb::DistanceMatrix<T> dPrime(n);
      for (size_t i = 0; i < n; ++i)
      {
        for (size_t j = i + 1; j < n; ++j)
        {
          const double base = 0.1 + unit(rng);
          d(i, j) = static_cast<T>(base);
          dPrime(i, j) = static_cast<T>(base * unit(rng));
        }
      }

      const auto bc = kernel_of(d, dPrime);
      ASSERT_EQ(bc.bars().size(), n - 1);

      std::vector<sb::ph::detail::MergeEdge<T>> dMerges;
      std::vector<sb::ph::detail::MergeEdge<T>> primeMerges;
      sb::ph::detail::mst_merge_order(d, n, dMerges);
      sb::ph::detail::mst_merge_order(dPrime, n, primeMerges);

      std::vector<T> births;
      std::vector<T> deaths;
      for (const auto &bar : bc.bars())
      {
        births.push_back(bar.birth);
        deaths.push_back(bar.death);
      }
      std::sort(births.begin(), births.end());
      std::sort(deaths.begin(), deaths.end());

      for (size_t i = 0; i + 1 < n; ++i)
      {
        EXPECT_EQ(births[i], primeMerges[i].mergeDist);
        EXPECT_EQ(deaths[i], dMerges[i].mergeDist);
      }
    }
  }

  TYPED_TEST(HomologicalKernelTest, NonDominatedMetricsThrow)
  {
    using T = TypeParam;

    // d' > d pointwise: the first d' merge is born at 2 but dies at 1.
    std::vector<std::vector<T>> d = {
        {T(0), T(1), T(1)},
        {T(1), T(0), T(1)},
        {T(1), T(1), T(0)},
    };
    std::vector<std::vector<T>> dPrime = {
        {T(0), T(2), T(2)},
        {T(2), T(0), T(2)},
        {T(2), T(2), T(0)},
    };

    EXPECT_THROW(kernel_of(make_distmat(d), make_distmat(dPrime)), std::runtime_error);
  }

  TYPED_TEST(HomologicalKernelTest, RoundoffLevelDominationViolationIsClamped)
  {
    using T = TypeParam;

    // d is one ULP below d' on the only pair: mathematically d = d' (an empty
    // bar), and the discrepancy is pure roundoff from computing the two sides
    // through different arithmetic. This must clamp to [w, w), not throw.
    const T w = T(1.5);
    const T justBelow = std::nextafter(w, T(0));
    std::vector<std::vector<T>> d = {
        {T(0), justBelow},
        {justBelow, T(0)},
    };
    std::vector<std::vector<T>> dPrime = {
        {T(0), w},
        {w, T(0)},
    };

    expect_kernel_bars<T>(d, dPrime, {{w, w}});
  }

  TYPED_TEST(HomologicalKernelTest, TwoPointsGiveSingleBar)
  {
    using T = TypeParam;

    // Minimal nontrivial case: one merge, one bar — the sweep at its smallest
    // size (a single d-MST edge).
    std::vector<std::vector<T>> d = {
        {T(0), T(5)},
        {T(5), T(0)},
    };
    std::vector<std::vector<T>> dPrime = {
        {T(0), T(2)},
        {T(2), T(0)},
    };

    expect_kernel_bars<T>(d, dPrime, {{T(2), T(5)}});
  }

  TYPED_TEST(HomologicalKernelTest, TrivialInputsGiveEmptyBarcode)
  {
    using T = TypeParam;

    EXPECT_TRUE(kernel_of(sb::DistanceMatrix<T>(1), sb::DistanceMatrix<T>(1)).bars().empty());
    EXPECT_TRUE(kernel_of(sb::DistanceMatrix<T>(0), sb::DistanceMatrix<T>(0)).bars().empty());
  }

  // ============================================================================
  // Per-instance wrappers (tensor cell level)
  // ============================================================================

  TYPED_TEST(HomologicalKernelTest, DistmatWrapperComputesAndRejectsRaggedElements)
  {
    using T = TypeParam;

    sb::Tensor<sb::DistanceMatrix<T>> input({1});
    sb::Tensor<sb::DistanceMatrix<T>> inputPrime({1});
    sb::Tensor<sb::ph::Barcode<T>> ret({1});

    auto d = line_metric<T>();
    auto dPrime = d;
    for (auto &row : dPrime)
    {
      for (auto &v : row)
      {
        v /= T(2);
      }
    }
    input({0}) = make_distmat<T>(d);
    inputPrime({0}) = make_distmat<T>(dPrime);

    sb::ph::detail::homological_kernel_distmat_single_impl(input, inputPrime, ret, {0});
    EXPECT_EQ(ret({0}).bars().size(), 3U);

    // Same outer shape but ragged element sizes must throw.
    inputPrime({0}) = sb::DistanceMatrix<T>(3);
    EXPECT_THROW(
        sb::ph::detail::homological_kernel_distmat_single_impl(input, inputPrime, ret, {0}), std::runtime_error);
  }

  TYPED_TEST(HomologicalKernelTest, PcloudWrapperRejectsMismatchedElementShapes)
  {
    using T = TypeParam;

    sb::Tensor<sb::PointCloud<T>> input({1});
    sb::Tensor<sb::PointCloud<T>> inputPrime({1});
    sb::Tensor<sb::ph::Barcode<T>> ret({1});

    input({0}) = make_pcloud<T>({{T(0), T(0)}, {T(1), T(0)}});
    inputPrime({0}) = make_pcloud<T>({{T(0)}, {T(1)}}); // same n, different dim

    EXPECT_THROW(
        sb::ph::detail::homological_kernel_pcloud_single_impl(input, inputPrime, ret, {0}), std::runtime_error);
  }

  TYPED_TEST(HomologicalKernelTest, PcloudWrapperRejectsRankOneElements)
  {
    using T = TypeParam;

    sb::Tensor<sb::PointCloud<T>> input({1});
    sb::Tensor<sb::PointCloud<T>> inputPrime({1});
    sb::Tensor<sb::ph::Barcode<T>> ret({1});

    // A rank-1 "cloud" (a bare vector, no coordinate axis) is not a valid
    // (n, dim) point cloud and must be rejected.
    sb::PointCloud<T> vec(std::vector<size_t>{3});
    vec({0}) = T(1);
    vec({1}) = T(2);
    vec({2}) = T(3);
    input({0}) = vec;
    inputPrime({0}) = vec;

    EXPECT_THROW(
        sb::ph::detail::homological_kernel_pcloud_single_impl(input, inputPrime, ret, {0}), std::runtime_error);

    // Empty rank-1 shapes are rejected the same way: the rank check runs
    // before any degeneracy handling, so validation does not depend on
    // whether the malformed cloud happens to be empty.
    input({0}) = sb::PointCloud<T>(std::vector<size_t>{0});
    inputPrime({0}) = sb::PointCloud<T>(std::vector<size_t>{0});
    EXPECT_THROW(
        sb::ph::detail::homological_kernel_pcloud_single_impl(input, inputPrime, ret, {0}), std::runtime_error);
  }

  TYPED_TEST(HomologicalKernelTest, PcloudWrapperTreatsZeroDimCloudsAsCoincidentPoints)
  {
    using T = TypeParam;

    sb::Tensor<sb::PointCloud<T>> input({1});
    sb::Tensor<sb::PointCloud<T>> inputPrime({1});
    sb::Tensor<sb::ph::Barcode<T>> ret({1});

    // (n, 0): n points with no coordinates. All points coincide, inducing the
    // all-zero metric — n-1 zero-length bars, matching the distmat route on
    // the equivalent all-zero matrix and the documented n-1 bar count.
    input({0}) = sb::PointCloud<T>(std::vector<size_t>{3, 0});
    inputPrime({0}) = sb::PointCloud<T>(std::vector<size_t>{3, 0});

    sb::ph::detail::homological_kernel_pcloud_single_impl(input, inputPrime, ret, {0});
    expect_bars<T>(ret({0}), {{T(0), T(0)}, {T(0), T(0)}});

    // (0, dim): no points at all — empty barcode.
    input({0}) = sb::PointCloud<T>(std::vector<size_t>{0, 2});
    inputPrime({0}) = sb::PointCloud<T>(std::vector<size_t>{0, 2});
    sb::ph::detail::homological_kernel_pcloud_single_impl(input, inputPrime, ret, {0});
    EXPECT_TRUE(ret({0}).bars().empty());
  }

} // namespace
