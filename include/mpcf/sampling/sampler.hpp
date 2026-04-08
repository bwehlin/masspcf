/*
* Copyright 2024-2026 Bjorn Wehlin
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef MPCF_SAMPLING_SAMPLER_H
#define MPCF_SAMPLING_SAMPLER_H

#include "distribution.hpp"
#include "sampling_result.hpp"
#include "kdtree.hpp"
#include "../tensor.hpp"
#include "../walk.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mpcf::sampling
{

  /// Sentinel value indicating no valid sample was drawn.
  constexpr size_t no_sample = static_cast<size_t>(-1);

  /// Default escalation stages for adaptive correction clamping.
  constexpr std::array<double, 4> default_stages_array = {0.0, 0.1, 0.5, 1.0};

  /// Default number of consecutive rejections before escalating to the next stage.
  constexpr size_t default_escalation_threshold = 100;

  /// Default maximum number of sample_one attempts per requested sample.
  constexpr size_t default_max_attempts = 1000;

  namespace detail
  {
    /// Compute the minimum squared distance from a point to an axis-aligned bounding box.
    template <typename T>
    T min_sq_dist_to_bbox(const std::vector<T>& point, const std::vector<T>& bbox_lo, const std::vector<T>& bbox_hi)
    {
      T dist = T(0);
      for (size_t i = 0; i < point.size(); ++i)
      {
        if (point[i] < bbox_lo[i])
        {
          T d = bbox_lo[i] - point[i];
          dist += d * d;
        }
        else if (point[i] > bbox_hi[i])
        {
          T d = point[i] - bbox_hi[i];
          dist += d * d;
        }
      }
      return dist;
    }

    /// Compute the maximum squared distance from a point to an axis-aligned bounding box.
    template <typename T>
    T max_sq_dist_to_bbox(const std::vector<T>& point, const std::vector<T>& bbox_lo, const std::vector<T>& bbox_hi)
    {
      T dist = T(0);
      for (size_t i = 0; i < point.size(); ++i)
      {
        T d_lo = point[i] - bbox_lo[i];
        T d_hi = point[i] - bbox_hi[i];
        T d = std::max(std::abs(d_lo), std::abs(d_hi));
        dist += d * d;
      }
      return dist;
    }

    /// Precompute subtree point counts for every node in the KD-tree.
    template <typename T>
    using SubtreeSizeMap = std::unordered_map<const typename KDTree<T>::Node*, size_t>;

    template <typename T>
    size_t build_subtree_sizes(const typename KDTree<T>::Node* node, SubtreeSizeMap<T>& sizes)
    {
      if (node->child1 == nullptr) // leaf
      {
        size_t count = node->node_type.lr.right - node->node_type.lr.left;
        sizes[node] = count;
        return count;
      }
      size_t count = build_subtree_sizes<T>(node->child1, sizes)
                   + build_subtree_sizes<T>(node->child2, sizes);
      sizes[node] = count;
      return count;
    }

    struct VantageSampleResult
    {
      std::vector<size_t> accepted;
      size_t total_attempts;
      bool was_biased;
    };

    /// Draw k samples using a draw_one callable, handling with/without replacement
    /// and adaptive escalation.
    ///
    /// @param draw_one  Callable with signature size_t(T min_correction).
    ///                  Returns a point index, or no_sample on rejection.
    template <typename T, typename DrawOneFn>
    VantageSampleResult draw_k_samples(
        DrawOneFn&& draw_one, size_t k, bool replace,
        const std::vector<T>& stages, size_t escalation_threshold, size_t max_attempts)
    {
      std::vector<size_t> accepted;
      accepted.reserve(k);

      size_t stage = 0;
      size_t consecutive_rejections = 0;
      size_t total_attempts = 0;
      std::unordered_set<size_t> seen;

      for (size_t s = 0; s < k; ++s)
      {
        bool got_sample = false;
        for (size_t attempt = 0; attempt < max_attempts && !got_sample; ++attempt)
        {
          ++total_attempts;
          size_t pidx = draw_one(stages[stage]);

          if (pidx != no_sample && (replace || seen.find(pidx) == seen.end()))
          {
            if (!replace) seen.insert(pidx);
            accepted.push_back(pidx);
            consecutive_rejections = 0;
            got_sample = true;
          }
          else
          {
            ++consecutive_rejections;
            if (consecutive_rejections >= escalation_threshold && stage + 1 < stages.size())
            {
              ++stage;
              consecutive_rejections = 0;
            }
          }
        }
      }

      return {std::move(accepted), total_attempts, stage > 0};
    }

    /// Common validation and result-building for both sampler types.
    template <typename T, typename MakeDrawFn, typename EngineT>
    SamplingResult<T> run_sampling(
        const PointCloud<T>& source,
        const PointCloud<T>& vantage,
        size_t k,
        bool replace,
        const RandomGenerator<EngineT>& gen,
        Executor& exec,
        T radius,
        const std::vector<T>& stages,
        size_t escalation_threshold,
        size_t max_attempts,
        MakeDrawFn&& make_draw)
    {
      size_t nVantage = vantage.shape()[0];
      T radius_sq = radius * radius;

      Tensor<uint64_t> allIndices({nVantage, k});
      std::fill(allIndices.data(), allIndices.data() + nVantage * k, no_sample);

      std::vector<size_t> acceptedCounts(nVantage, 0);
      std::vector<size_t> attemptCounts(nVantage, 0);
      std::vector<bool> biasedFlags(nVantage, false);

      Tensor<T> walkShape({nVantage});

      exec.cpu()->wait_for_all();

      mpcf::parallel_walk(walkShape, gen,
          [&vantage, &allIndices, &acceptedCounts, &attemptCounts, &biasedFlags,
           &stages, &make_draw, k, radius_sq, replace, escalation_threshold, max_attempts]
          (const std::vector<size_t>& idx, auto& engine)
      {
        size_t vi = idx[0];

        auto draw = make_draw(vantage, vi, radius_sq, engine);
        auto result = draw_k_samples<T>(draw, k, replace, stages, escalation_threshold, max_attempts);

        for (size_t s = 0; s < result.accepted.size(); ++s)
        {
          allIndices({vi, s}) = result.accepted[s];
        }
        acceptedCounts[vi] = result.accepted.size();
        attemptCounts[vi] = result.total_attempts;
        biasedFlags[vi] = result.was_biased;
      }, exec);

      // Build diagnostics
      SamplingDiagnostics diagnostics;
      diagnostics.acceptance_rate.resize(nVantage);
      diagnostics.total_attempts = std::move(attemptCounts);
      diagnostics.biased = std::move(biasedFlags);
      for (size_t i = 0; i < nVantage; ++i)
      {
        diagnostics.acceptance_rate[i] = diagnostics.total_attempts[i] > 0
            ? static_cast<double>(acceptedCounts[i]) / static_cast<double>(diagnostics.total_attempts[i])
            : 0.0;
      }

      // Compact indices
      bool allFull = true;
      for (size_t i = 0; i < nVantage; ++i)
      {
        if (acceptedCounts[i] != k)
        {
          allFull = false;
          break;
        }
      }

      if (allFull)
      {
        return SamplingResult<T>{
            IndexedPointCloudCollection<T>(source, std::move(allIndices)),
            std::move(diagnostics)};
      }

      size_t maxAccepted = *std::max_element(acceptedCounts.begin(), acceptedCounts.end());
      if (maxAccepted == 0)
      {
        Tensor<uint64_t> emptyIndices({nVantage, 0_uz});
        return SamplingResult<T>{
            IndexedPointCloudCollection<T>(source, std::move(emptyIndices)),
            std::move(diagnostics)};
      }

      Tensor<uint64_t> compactIndices({nVantage, maxAccepted});
      for (size_t i = 0; i < nVantage; ++i)
      {
        for (size_t j = 0; j < maxAccepted; ++j)
        {
          compactIndices({i, j}) = (j < acceptedCounts[i]) ? allIndices({i, j}) : 0;
        }
      }

      return SamplingResult<T>{
          IndexedPointCloudCollection<T>(source, std::move(compactIndices)),
          std::move(diagnostics)};
    }

    template <typename T>
    void validate_vantage(const PointCloud<T>& vantage, size_t expected_dim)
    {
      if (vantage.rank() != 2)
      {
        throw std::invalid_argument("vantage must have rank 2 (M x D)");
      }
      if (vantage.shape()[1] != expected_dim)
      {
        throw std::invalid_argument("source and vantage must have same dimensionality");
      }
    }

  } // namespace detail


  /// Brute-force distance-weighted sampler.
  ///
  /// Computes all N distances per vantage point and samples via CDF inversion.
  /// O(Nk) per vantage — no tree build cost, but linear in cloud size.
  template <typename T>
  class NaiveSampler
  {
  public:
    explicit NaiveSampler(PointCloud<T> source, Executor& exec = default_executor())
      : m_source(std::move(source))
      , m_dim(m_source.shape()[1])
      , m_adaptor(m_source)
      , m_exec(&exec)
    {
      if (m_source.rank() != 2)
      {
        throw std::invalid_argument("source must have rank 2 (N x D)");
      }
    }

    NaiveSampler(const NaiveSampler&) = delete;
    NaiveSampler& operator=(const NaiveSampler&) = delete;

    size_t dim() const { return m_dim; }
    size_t n_points() const { return m_source.shape()[0]; }
    const PointCloud<T>& source() const { return m_source; }

    template <typename Dist, typename EngineT>
    SamplingResult<T> sample(
        const PointCloud<T>& vantage,
        size_t k,
        const Dist& dist,
        bool replace,
        const RandomGenerator<EngineT>& gen,
        T radius = std::numeric_limits<T>::infinity(),
        const std::vector<T>& stages = {T(0), T(0.1), T(0.5), T(1.0)},
        size_t escalation_threshold = default_escalation_threshold,
        size_t max_attempts = default_max_attempts) const
    {
      static_assert(WeightFunction<Dist, T>, "Dist must satisfy WeightFunction concept");
      detail::validate_vantage(vantage, m_dim);

      if (stages.empty())
      {
        throw std::invalid_argument("stages must not be empty");
      }

      return detail::run_sampling(m_source, vantage, k, replace, gen, *m_exec,
          radius, stages, escalation_threshold, max_attempts,
          [this, &dist](const PointCloud<T>& v, size_t vi, T radius_sq, auto& engine)
      {
        return make_draw(v, vi, dist, radius_sq, engine);
      });
    }

  private:
    PointCloud<T> m_source;
    size_t m_dim;
    PointCloudAdaptor<T> m_adaptor;
    Executor* m_exec;

    template <typename Dist, typename EngineT>
    auto make_draw(
        const PointCloud<T>& vantage, size_t vi, const Dist& dist,
        T radius_sq, EngineT& engine) const
    {
      size_t n = n_points();
      std::vector<T> cdf(n);
      T total_weight = T(0);

      for (size_t i = 0; i < n; ++i)
      {
        T sq_dist = T(0);
        for (size_t d = 0; d < m_dim; ++d)
        {
          T diff = vantage({vi, d}) - m_adaptor.cloud({i, d});
          sq_dist += diff * diff;
        }

        T w = (sq_dist > radius_sq) ? T(0) : dist(std::sqrt(sq_dist));
        total_weight += w;
        cdf[i] = total_weight;
      }

      return [cdf = std::move(cdf), total_weight, n, &engine](T) -> size_t
      {
        if (total_weight <= T(0)) return no_sample;

        std::uniform_real_distribution<T> uniform01(T(0), T(1));
        T u = uniform01(engine) * total_weight;
        auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        size_t idx = static_cast<size_t>(std::distance(cdf.begin(), it));
        return (idx < n) ? idx : n - 1;
      };
    }
  };


  /// KD-tree importance sampler.
  ///
  /// Builds a KD-tree over the source cloud once (using parallel construction),
  /// then supports O(log N) per-sample cost via tree-based importance sampling
  /// with adaptive escalation.
  template <typename T>
  class TreeSampler
  {
  public:
    explicit TreeSampler(PointCloud<T> source, Executor& exec = default_executor())
      : m_source(std::move(source))
      , m_dim(m_source.shape()[1])
      , m_adaptor(m_source)
      , m_exec(&exec)
    {
      if (m_source.rank() != 2)
      {
        throw std::invalid_argument("source must have rank 2 (N x D)");
      }
    }

    TreeSampler(const TreeSampler&) = delete;
    TreeSampler& operator=(const TreeSampler&) = delete;

    size_t dim() const { return m_dim; }
    size_t n_points() const { return m_source.shape()[0]; }
    const PointCloud<T>& source() const { return m_source; }

    template <typename Dist, typename EngineT>
    SamplingResult<T> sample(
        const PointCloud<T>& vantage,
        size_t k,
        const Dist& dist,
        bool replace,
        const RandomGenerator<EngineT>& gen,
        T radius = std::numeric_limits<T>::infinity(),
        const std::vector<T>& stages = {T(0), T(0.1), T(0.5), T(1.0)},
        size_t escalation_threshold = default_escalation_threshold,
        size_t max_attempts = default_max_attempts)
    {
      static_assert(WeightFunction<Dist, T>, "Dist must satisfy WeightFunction concept");
      detail::validate_vantage(vantage, m_dim);

      if (stages.empty())
      {
        throw std::invalid_argument("stages must not be empty");
      }

      ensure_tree_built();

      return detail::run_sampling(m_source, vantage, k, replace, gen, *m_exec,
          radius, stages, escalation_threshold, max_attempts,
          [this, &dist](const PointCloud<T>& v, size_t vi, T radius_sq, auto& engine)
      {
        return make_draw(v, vi, dist, radius_sq, engine);
      });
    }

  private:
    PointCloud<T> m_source;
    size_t m_dim;
    PointCloudAdaptor<T> m_adaptor;
    Executor* m_exec;
    std::unique_ptr<KDTree<T>> m_tree;
    detail::SubtreeSizeMap<T> m_subtreeSizes;

    void ensure_tree_built()
    {
      if (m_tree) return;

      size_t n_threads = m_exec->cpu()->num_workers();

      // Lock up the executor's workers while nanoflann builds with its own
      // threads, so nothing else competes for the same cores.
      m_exec->cpu()->wait_for_all();
      std::promise<void> gate;
      auto shared_gate = gate.get_future().share();
      tf::Taskflow blocker;
      for (size_t i = 0; i < n_threads; ++i)
      {
        blocker.emplace([shared_gate]() { shared_gate.wait(); });
      }
      auto blocker_future = m_exec->cpu()->run(std::move(blocker));

      m_tree = std::make_unique<KDTree<T>>(
          static_cast<int>(m_dim), m_adaptor,
          nanoflann::KDTreeSingleIndexAdaptorParams{10, nanoflann::KDTreeSingleIndexAdaptorFlags::None, n_threads});
      m_tree->buildIndex();

      gate.set_value();
      blocker_future.wait();

      detail::build_subtree_sizes<T>(m_tree->root_node_, m_subtreeSizes);
    }

    template <typename Dist, typename EngineT>
    auto make_draw(
        const PointCloud<T>& vantage, size_t vi, const Dist& dist,
        T radius_sq, EngineT& engine) const
    {
      std::vector<T> vpt(m_dim);
      for (size_t d = 0; d < m_dim; ++d)
      {
        vpt[d] = vantage({vi, d});
      }

      return [this, &dist, radius_sq, &engine,
              vpt = std::move(vpt),
              bbox_lo = std::vector<T>(m_dim),
              bbox_hi = std::vector<T>(m_dim)]
             (T min_correction) mutable -> size_t
      {
        return sample_one(vpt, dist, radius_sq, engine,
                          bbox_lo, bbox_hi, min_correction);
      };
    }

    template <typename Dist, typename EngineT>
    size_t sample_one(
        const std::vector<T>& vantage,
        const Dist& dist,
        T radius_sq,
        EngineT& engine,
        std::vector<T>& bbox_lo,
        std::vector<T>& bbox_hi,
        T min_correction) const
    {
      using Node = typename KDTree<T>::Node;

      std::uniform_real_distribution<T> uniform01(T(0), T(1));

      for (size_t i = 0; i < m_dim; ++i)
      {
        bbox_lo[i] = m_tree->root_bbox_[i].low;
        bbox_hi[i] = m_tree->root_bbox_[i].high;
      }

      const Node* node = m_tree->root_node_;

      // Cumulative correction factor for unbiased sampling.
      // At each internal node, children's refined bounding boxes give tighter
      // weight bounds than the parent's estimate. The ratio S/w_self (where
      // S = W_L + W_R and w_self is the bound the parent used for this node)
      // captures the tightening. Accumulating this product ensures the final
      // acceptance probability is g(d) / S_root for every point, removing
      // the path-dependent bias.
      T correction = T(1);
      T w_self = T(0);

      while (node->child1 != nullptr)
      {
        auto cutfeat = node->node_type.sub.divfeat;
        T divlow = static_cast<T>(node->node_type.sub.divlow);
        T divhigh = static_cast<T>(node->node_type.sub.divhigh);

        T saved_hi = bbox_hi[cutfeat];
        T saved_lo = bbox_lo[cutfeat];

        bbox_hi[cutfeat] = divlow;
        T left_min_sq = detail::min_sq_dist_to_bbox(vantage, bbox_lo, bbox_hi);
        T left_max_sq = detail::max_sq_dist_to_bbox(vantage, bbox_lo, bbox_hi);
        bbox_hi[cutfeat] = saved_hi;

        bbox_lo[cutfeat] = divhigh;
        T right_min_sq = detail::min_sq_dist_to_bbox(vantage, bbox_lo, bbox_hi);
        T right_max_sq = detail::max_sq_dist_to_bbox(vantage, bbox_lo, bbox_hi);
        bbox_lo[cutfeat] = saved_lo;

        T left_d_min = std::sqrt(left_min_sq);
        T left_d_max = std::sqrt(left_max_sq);
        T right_d_min = std::sqrt(right_min_sq);
        T right_d_max = std::sqrt(right_max_sq);

        T left_count = static_cast<T>(m_subtreeSizes.at(node->child1));
        T right_count = static_cast<T>(m_subtreeSizes.at(node->child2));

        T left_w = T(0);
        T right_w = T(0);

        if (left_min_sq <= radius_sq)
        {
          left_w = left_count * dist.max_in_range(left_d_min, left_d_max);
        }

        if (right_min_sq <= radius_sq)
        {
          right_w = right_count * dist.max_in_range(right_d_min, right_d_max);
        }

        T total_w = left_w + right_w;
        if (total_w <= T(0))
        {
          return no_sample;
        }

        if (w_self > T(0))
        {
          correction *= total_w / w_self;
          correction = std::max(correction, min_correction);
        }

        T u = uniform01(engine);
        if (u < left_w / total_w)
        {
          node = node->child1;
          bbox_hi[cutfeat] = divlow;
          w_self = left_w;
        }
        else
        {
          node = node->child2;
          bbox_lo[cutfeat] = divhigh;
          w_self = right_w;
        }
      }

      auto left = node->node_type.lr.left;
      auto right = node->node_type.lr.right;
      size_t n_leaf = right - left;

      std::uniform_int_distribution<size_t> leaf_dist(0, n_leaf - 1);
      size_t leaf_idx = leaf_dist(engine);
      size_t point_idx = m_tree->vAcc_[left + leaf_idx];

      T sq_dist = T(0);
      for (size_t i = 0; i < m_dim; ++i)
      {
        T d = vantage[i] - m_adaptor.cloud({point_idx, i});
        sq_dist += d * d;
      }

      if (sq_dist > radius_sq)
      {
        return no_sample;
      }

      T actual_dist = std::sqrt(sq_dist);
      T actual_weight = dist(actual_dist);

      // Leaf-level accept/reject: multiplying by the cumulative correction
      // makes the acceptance probability g(d) / S_root for every point,
      // so accepted samples are exactly proportional to g(d).
      T leaf_d_min = std::sqrt(detail::min_sq_dist_to_bbox(vantage, bbox_lo, bbox_hi));
      T leaf_d_max = std::sqrt(detail::max_sq_dist_to_bbox(vantage, bbox_lo, bbox_hi));
      T leaf_max_weight = dist.max_in_range(leaf_d_min, leaf_d_max);

      if (leaf_max_weight <= T(0))
      {
        return no_sample;
      }

      T leaf_correction = correction;
      if (w_self > T(0))
      {
        T leaf_total = static_cast<T>(n_leaf) * leaf_max_weight;
        leaf_correction *= leaf_total / w_self;
        leaf_correction = std::max(leaf_correction, min_correction);
      }

      T accept_prob = (actual_weight / leaf_max_weight) * leaf_correction;
      if (uniform01(engine) < accept_prob)
      {
        return point_idx;
      }

      return no_sample;
    }
  };


}

#endif
