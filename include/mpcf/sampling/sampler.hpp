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
#include <limits>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mpcf::sampling
{

  namespace detail
  {
    /// Compute the minimum squared distance from a point to an axis-aligned bounding box.
    template <typename T>
    T min_sq_dist_to_bbox(const T* point, const std::vector<T>& bbox_lo, const std::vector<T>& bbox_hi, size_t dim)
    {
      T dist = T(0);
      for (size_t i = 0; i < dim; ++i)
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
    T max_sq_dist_to_bbox(const T* point, const std::vector<T>& bbox_lo, const std::vector<T>& bbox_hi, size_t dim)
    {
      T dist = T(0);
      for (size_t i = 0; i < dim; ++i)
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
  }

  /// Precomputed state for distance-weighted importance sampling from a point cloud.
  ///
  /// Builds a KD-tree over the source cloud once, then supports efficient
  /// repeated sampling with different vantage points and weight functions.
  template <typename T>
  class DistanceWeightedSampler
  {
  public:
    explicit DistanceWeightedSampler(PointCloud<T> source)
      : m_source(std::move(source))
      , m_dim(m_source.shape()[1])
      , m_adaptor(m_source)
    {
      if (m_source.rank() != 2)
      {
        throw std::invalid_argument("source must have rank 2 (N x D)");
      }

      m_tree = std::make_unique<KDTree<T>>(
          static_cast<int>(m_dim), m_adaptor,
          nanoflann::KDTreeSingleIndexAdaptorParams{10});
      m_tree->buildIndex();

      detail::build_subtree_sizes<T>(m_tree->root_node_, m_subtreeSizes);
    }

    DistanceWeightedSampler(const DistanceWeightedSampler&) = delete;
    DistanceWeightedSampler& operator=(const DistanceWeightedSampler&) = delete;

    size_t dim() const { return m_dim; }
    size_t n_points() const { return m_source.shape()[0]; }

    /// Default escalation stages for adaptive correction clamping.
    static constexpr std::array<double, 4> default_stages = {0.0, 0.1, 0.5, 1.0};

    /// Default number of consecutive rejections before escalating to the next stage.
    static constexpr size_t default_escalation_threshold = 100;

    /// Default maximum number of sample_one attempts per requested sample.
    static constexpr size_t default_max_attempts = 1000;

    /// Sample k points around each vantage point, weighted by a distance distribution.
    ///
    /// Uses adaptive escalation: starts with exact (unbiased) sampling and
    /// automatically introduces bounded bias when acceptance rates are too low.
    ///
    /// @param vantage   Vantage points of shape (M, D).
    /// @param k         Number of samples per vantage point.
    /// @param dist      Weight function (must satisfy WeightFunction concept).
    /// @param replace   If true, sample with replacement; otherwise without.
    /// @param gen       Random generator for deterministic seeding.
    /// @param exec      Executor for parallel dispatch.
    /// @param radius    Optional ball radius (default: infinity = unrestricted).
    /// @param stages    Escalation stages for correction clamping (default: {0, 0.1, 0.5, 1.0}).
    /// @param escalation_threshold  Consecutive rejections before escalating (default: 100).
    /// @param max_attempts  Maximum sample_one calls per requested sample (default: 1000).
    template <typename Dist, typename EngineT>
    SamplingResult<T> sample(
        const PointCloud<T>& vantage,
        size_t k,
        const Dist& dist,
        bool replace,
        const RandomGenerator<EngineT>& gen,
        Executor& exec,
        T radius = std::numeric_limits<T>::infinity(),
        const std::vector<T>& stages = {T(0), T(0.1), T(0.5), T(1.0)},
        size_t escalation_threshold = default_escalation_threshold,
        size_t max_attempts = default_max_attempts) const
    {
      static_assert(WeightFunction<Dist, T>, "Dist must satisfy WeightFunction concept");

      if (vantage.rank() != 2)
      {
        throw std::invalid_argument("vantage must have rank 2 (M x D)");
      }
      if (vantage.shape()[1] != m_dim)
      {
        throw std::invalid_argument("source and vantage must have same dimensionality");
      }
      if (stages.empty())
      {
        throw std::invalid_argument("stages must not be empty");
      }

      size_t nVantage = vantage.shape()[0];
      T radius_sq = radius * radius;

      Tensor<uint64_t> allIndices({nVantage, k});

      constexpr size_t sentinel = static_cast<size_t>(-1);
      std::fill(allIndices.data(), allIndices.data() + nVantage * k, sentinel);

      std::vector<size_t> acceptedCounts(nVantage, 0);
      std::vector<size_t> attemptCounts(nVantage, 0);
      std::vector<bool> biasedFlags(nVantage, false);

      Tensor<T> dummy({nVantage});

      mpcf::parallel_walk(dummy, gen,
          [this, &vantage, &dist, &allIndices, &acceptedCounts, &attemptCounts, &biasedFlags,
           &stages, k, radius_sq, replace, max_attempts, escalation_threshold]
          (const std::vector<size_t>& idx, auto& engine)
      {
        size_t vantageIdx = idx[0];

        std::vector<T> vpt(m_dim);
        for (size_t d = 0; d < m_dim; ++d)
        {
          vpt[d] = vantage({vantageIdx, d});
        }

        // Working buffers reused across all sample attempts for this vantage
        std::vector<T> bbox_lo(m_dim);
        std::vector<T> bbox_hi(m_dim);
        std::uniform_real_distribution<T> uniform01(T(0), T(1));

        std::vector<size_t> accepted;
        accepted.reserve(k);

        size_t stage = 0;
        size_t consecutive_rejections = 0;
        size_t total_attempts = 0;

        if (replace)
        {
          for (size_t s = 0; s < k; ++s)
          {
            bool got_sample = false;
            for (size_t attempt = 0; attempt < max_attempts && !got_sample; ++attempt)
            {
              ++total_attempts;
              size_t pidx = sample_one(
                  vpt.data(), dist, radius_sq, engine, uniform01,
                  bbox_lo, bbox_hi, stages[stage]);
              if (pidx != static_cast<size_t>(-1))
              {
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
        }
        else
        {
          std::unordered_set<size_t> seen;

          for (size_t s = 0; s < k; ++s)
          {
            bool got_sample = false;
            for (size_t attempt = 0; attempt < max_attempts && !got_sample; ++attempt)
            {
              ++total_attempts;
              size_t pidx = sample_one(
                  vpt.data(), dist, radius_sq, engine, uniform01,
                  bbox_lo, bbox_hi, stages[stage]);
              if (pidx != static_cast<size_t>(-1) && seen.find(pidx) == seen.end())
              {
                seen.insert(pidx);
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
        }

        for (size_t s = 0; s < accepted.size(); ++s)
        {
          allIndices({vantageIdx, s}) = accepted[s];
        }
        acceptedCounts[vantageIdx] = accepted.size();
        attemptCounts[vantageIdx] = total_attempts;
        biasedFlags[vantageIdx] = (stage > 0);
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
            IndexedPointCloudCollection<T>(m_source, std::move(allIndices)),
            std::move(diagnostics)};
      }

      size_t maxAccepted = *std::max_element(acceptedCounts.begin(), acceptedCounts.end());
      if (maxAccepted == 0)
      {
        Tensor<uint64_t> emptyIndices({nVantage, 0_uz});
        return SamplingResult<T>{
            IndexedPointCloudCollection<T>(m_source, std::move(emptyIndices)),
            std::move(diagnostics)};
      }

      Tensor<uint64_t> compactIndices({nVantage, maxAccepted});
      for (size_t i = 0; i < nVantage; ++i)
      {
        for (size_t j = 0; j < maxAccepted; ++j)
        {
          if (j < acceptedCounts[i])
          {
            compactIndices({i, j}) = allIndices({i, j});
          }
          else
          {
            compactIndices({i, j}) = 0;
          }
        }
      }

      return SamplingResult<T>{
          IndexedPointCloudCollection<T>(m_source, std::move(compactIndices)),
          std::move(diagnostics)};
    }

  private:
    PointCloud<T> m_source;
    size_t m_dim;
    PointCloudAdaptor<T> m_adaptor;
    std::unique_ptr<KDTree<T>> m_tree;
    detail::SubtreeSizeMap<T> m_subtreeSizes;

    /// Sample a single point index using tree-based importance sampling.
    /// Returns the sampled point index, or size_t(-1) if rejected.
    /// bbox_lo/bbox_hi are caller-owned working buffers, reset to root bbox each call.
    template <typename Dist, typename EngineT>
    size_t sample_one(
        const T* vantage,
        const Dist& dist,
        T radius_sq,
        EngineT& engine,
        std::uniform_real_distribution<T>& uniform01,
        std::vector<T>& bbox_lo,
        std::vector<T>& bbox_hi,
        T min_correction) const
    {
      using Node = typename KDTree<T>::Node;

      // Reset bounding box to root
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

      // Traverse tree top-down, modifying bbox in-place to avoid allocations
      while (node->child1 != nullptr)
      {
        auto cutfeat = node->node_type.sub.divfeat;
        T divlow = static_cast<T>(node->node_type.sub.divlow);
        T divhigh = static_cast<T>(node->node_type.sub.divhigh);

        T saved_hi = bbox_hi[cutfeat];
        T saved_lo = bbox_lo[cutfeat];

        // Compute left child distance range (bbox_hi[cutfeat] = divlow)
        bbox_hi[cutfeat] = divlow;
        T left_min_sq = detail::min_sq_dist_to_bbox(vantage, bbox_lo, bbox_hi, m_dim);
        T left_max_sq = detail::max_sq_dist_to_bbox(vantage, bbox_lo, bbox_hi, m_dim);
        bbox_hi[cutfeat] = saved_hi;

        // Compute right child distance range (bbox_lo[cutfeat] = divhigh)
        bbox_lo[cutfeat] = divhigh;
        T right_min_sq = detail::min_sq_dist_to_bbox(vantage, bbox_lo, bbox_hi, m_dim);
        T right_max_sq = detail::max_sq_dist_to_bbox(vantage, bbox_lo, bbox_hi, m_dim);
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
          return static_cast<size_t>(-1);
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

      // At a leaf node — pick a point uniformly among the leaf's points
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
        return static_cast<size_t>(-1);
      }

      T actual_dist = std::sqrt(sq_dist);
      T actual_weight = dist(actual_dist);

      // Leaf-level accept/reject with correction factor.
      // Multiplying by the cumulative correction makes the overall acceptance
      // probability g(d) / S_root — the same denominator for all points — so
      // accepted samples are exactly proportional to g(d).
      T leaf_d_min = std::sqrt(detail::min_sq_dist_to_bbox(vantage, bbox_lo, bbox_hi, m_dim));
      T leaf_d_max = std::sqrt(detail::max_sq_dist_to_bbox(vantage, bbox_lo, bbox_hi, m_dim));
      T leaf_max_weight = dist.max_in_range(leaf_d_min, leaf_d_max);

      if (leaf_max_weight <= T(0))
      {
        return static_cast<size_t>(-1);
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

      return static_cast<size_t>(-1);
    }
  };

}

#endif
