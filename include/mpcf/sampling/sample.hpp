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

#ifndef MPCF_SAMPLING_SAMPLE_H
#define MPCF_SAMPLING_SAMPLE_H

#include "distribution.hpp"
#include "indexed_point_cloud.hpp"
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

    /// Recursive tree traversal to sample a single point index using importance sampling.
    /// Returns the sampled point index (into the original point cloud), or size_t(-1) if rejected.
    template <typename T, typename Dist, typename EngineT>
    size_t sample_one_from_tree(
        const KDTree<T>& tree,
        const PointCloudAdaptor<T>& adaptor,
        const T* vantage,
        const Dist& dist,
        T radius_sq,
        EngineT& engine,
        size_t dim,
        const SubtreeSizeMap<T>& subtreeSizes,
        T min_correction)
    {
      using Node = typename KDTree<T>::Node;

      // Start with root node and root bounding box
      std::vector<T> root_lo(dim), root_hi(dim);
      for (size_t i = 0; i < dim; ++i)
      {
        root_lo[i] = tree.root_bbox_[i].low;
        root_hi[i] = tree.root_bbox_[i].high;
      }

      const Node* node = tree.root_node_;
      std::vector<T> bbox_lo = root_lo;
      std::vector<T> bbox_hi = root_hi;

      std::uniform_real_distribution<T> uniform01(T(0), T(1));

      // Cumulative correction factor for unbiased sampling.
      // At each internal node, children's refined bounding boxes give tighter
      // weight bounds than the parent's estimate. The ratio S/w_self (where
      // S = W_L + W_R and w_self is the bound the parent used for this node)
      // captures the tightening. Accumulating this product ensures the final
      // acceptance probability is g(d) / S_root for every point, removing
      // the path-dependent bias.
      T correction = T(1);
      T w_self = T(0); // weight bound assigned to this node by its parent

      // Traverse tree top-down
      while (node->child1 != nullptr) // not a leaf
      {
        auto cutfeat = node->node_type.sub.divfeat;
        T divlow = static_cast<T>(node->node_type.sub.divlow);
        T divhigh = static_cast<T>(node->node_type.sub.divhigh);

        // Left child bounding box: same as current but bbox_hi[cutfeat] = divlow
        std::vector<T> left_lo = bbox_lo;
        std::vector<T> left_hi = bbox_hi;
        left_hi[cutfeat] = divlow;

        // Right child bounding box: same as current but bbox_lo[cutfeat] = divhigh
        std::vector<T> right_lo = bbox_lo;
        std::vector<T> right_hi = bbox_hi;
        right_lo[cutfeat] = divhigh;

        // Compute distance ranges for each child
        T left_min_sq = min_sq_dist_to_bbox(vantage, left_lo, left_hi, dim);
        T left_max_sq = max_sq_dist_to_bbox(vantage, left_lo, left_hi, dim);
        T right_min_sq = min_sq_dist_to_bbox(vantage, right_lo, right_hi, dim);
        T right_max_sq = max_sq_dist_to_bbox(vantage, right_lo, right_hi, dim);

        T left_d_min = std::sqrt(left_min_sq);
        T left_d_max = std::sqrt(left_max_sq);
        T right_d_min = std::sqrt(right_min_sq);
        T right_d_max = std::sqrt(right_max_sq);

        // Point counts in each subtree (precomputed)
        T left_count = static_cast<T>(subtreeSizes.at(node->child1));
        T right_count = static_cast<T>(subtreeSizes.at(node->child2));

        // Compute weight upper bounds: n_points * max_density_in_range
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
          return static_cast<size_t>(-1); // No weight in this subtree
        }

        // Update correction: S / w_self (skip at root where w_self is 0)
        if (w_self > T(0))
        {
          correction *= total_w / w_self;
          correction = std::max(correction, min_correction);
        }

        // Choose left or right proportional to weight bounds
        T u = uniform01(engine);
        if (u < left_w / total_w)
        {
          node = node->child1;
          bbox_lo = std::move(left_lo);
          bbox_hi = std::move(left_hi);
          w_self = left_w;
        }
        else
        {
          node = node->child2;
          bbox_lo = std::move(right_lo);
          bbox_hi = std::move(right_hi);
          w_self = right_w;
        }
      }

      // At a leaf node — pick a point uniformly among the leaf's points
      auto left = node->node_type.lr.left;
      auto right = node->node_type.lr.right;
      size_t n_leaf = right - left;

      std::uniform_int_distribution<size_t> leaf_dist(0, n_leaf - 1);
      size_t leaf_idx = leaf_dist(engine);
      size_t point_idx = tree.vAcc_[left + leaf_idx];

      // Compute actual distance and accept/reject
      T sq_dist = T(0);
      for (size_t i = 0; i < dim; ++i)
      {
        T d = vantage[i] - adaptor.cloud({point_idx, i});
        sq_dist += d * d;
      }

      if (sq_dist > radius_sq)
      {
        return static_cast<size_t>(-1);
      }

      T actual_dist = std::sqrt(sq_dist);
      T actual_weight = dist(actual_dist);

      // Leaf-level accept/reject with correction factor.
      // The uncorrected acceptance would be g(d) / g_max(leaf_range), but this
      // ignores the tightening at intermediate nodes. Multiplying by the
      // cumulative correction makes the overall acceptance probability
      // g(d) / S_root — the same denominator for all points — so accepted
      // samples are exactly proportional to g(d).
      T leaf_d_min = std::sqrt(min_sq_dist_to_bbox(vantage, bbox_lo, bbox_hi, dim));
      T leaf_d_max = std::sqrt(max_sq_dist_to_bbox(vantage, bbox_lo, bbox_hi, dim));
      T leaf_max_weight = dist.max_in_range(leaf_d_min, leaf_d_max);

      if (leaf_max_weight <= T(0))
      {
        return static_cast<size_t>(-1);
      }

      // Apply leaf-level tightening (leaf bounds vs w_self from parent)
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
  }

  /// Sample k points around each vantage point, weighted by a distance distribution.
  ///
  /// Uses tree-based importance sampling: the KD-tree over the source cloud
  /// serves as a hierarchical proposal distribution. Works for any distribution
  /// shape (peaked, broad, heavy-tailed) without strategy switching.
  ///
  /// @param source    Source point cloud of shape (N, D).
  /// @param vantage   Vantage points of shape (M, D).
  /// @param k         Number of samples per vantage point.
  /// @param dist      Weight function (must satisfy WeightFunction concept).
  /// @param replace   If true, sample with replacement; otherwise without.
  /// @param gen       Random generator for deterministic seeding.
  /// @param exec      Executor for parallel dispatch.
  /// @param radius    Optional ball radius (default: infinity = unrestricted).
  /// @param min_correction  Minimum value for the cumulative correction factor.
  ///                  0.0 (default) gives exact unbiased sampling. Values in (0,1]
  ///                  clamp the correction, introducing bounded bias but improving
  ///                  acceptance rates for distributions with well-separated modes.
  ///                  1.0 disables the correction entirely.
  template <typename T, typename Dist, typename EngineT>
  IndexedPointCloudCollection<T> sample(
      const PointCloud<T>& source,
      const PointCloud<T>& vantage,
      size_t k,
      const Dist& dist,
      bool replace,
      const RandomGenerator<EngineT>& gen,
      Executor& exec,
      T radius = std::numeric_limits<T>::infinity(),
      T min_correction = T(0))
  {
    static_assert(WeightFunction<Dist, T>, "Dist must satisfy WeightFunction concept");

    if (source.rank() != 2)
    {
      throw std::invalid_argument("source must have rank 2 (N x D)");
    }
    if (vantage.rank() != 2)
    {
      throw std::invalid_argument("vantage must have rank 2 (M x D)");
    }

    size_t dim = source.shape()[1];
    if (vantage.shape()[1] != dim)
    {
      throw std::invalid_argument("source and vantage must have same dimensionality");
    }

    size_t nVantage = vantage.shape()[0];
    T radius_sq = radius * radius;

    // Build KD-tree over source
    PointCloudAdaptor<T> adaptor(source);
    KDTree<T> tree(static_cast<int>(dim), adaptor, nanoflann::KDTreeSingleIndexAdaptorParams{10});
    tree.buildIndex();

    // Precompute subtree sizes once
    detail::SubtreeSizeMap<T> subtreeSizes;
    detail::build_subtree_sizes<T>(tree.root_node_, subtreeSizes);

    // Maximum attempts per sample to avoid infinite loops
    constexpr size_t maxAttempts = 1000;

    // Allocate the index tensor (M, k) — filled in parallel
    Tensor<size_t> allIndices({nVantage, k});

    // Use a sentinel to mark unfilled slots (when fewer than k points are accepted)
    constexpr size_t sentinel = static_cast<size_t>(-1);
    for (size_t i = 0; i < nVantage * k; ++i)
    {
      allIndices.data()[i] = sentinel;
    }

    // We need to track how many samples each vantage actually accepted
    std::vector<size_t> acceptedCounts(nVantage, 0);

    // Create a dummy tensor for parallel_walk iteration over vantage points
    Tensor<T> dummy({nVantage});

    // Sample for each vantage point in parallel
    mpcf::parallel_walk(dummy, gen,
        [&tree, &adaptor, &vantage, &dist, &allIndices, &acceptedCounts, &subtreeSizes, k, dim, radius_sq, replace, maxAttempts, min_correction]
        (const std::vector<size_t>& idx, auto& engine)
    {
      size_t vantageIdx = idx[0];

      // Extract vantage point coordinates
      std::vector<T> vpt(dim);
      for (size_t d = 0; d < dim; ++d)
      {
        vpt[d] = vantage({vantageIdx, d});
      }

      // Collect accepted samples into a temporary buffer
      std::vector<size_t> accepted;
      accepted.reserve(k);

      if (replace)
      {
        for (size_t s = 0; s < k; ++s)
        {
          for (size_t attempt = 0; attempt < maxAttempts; ++attempt)
          {
            size_t pidx = detail::sample_one_from_tree(
                tree, adaptor, vpt.data(), dist, radius_sq, engine, dim, subtreeSizes, min_correction);
            if (pidx != static_cast<size_t>(-1))
            {
              accepted.push_back(pidx);
              break;
            }
          }
        }
      }
      else
      {
        std::unordered_set<size_t> seen;

        for (size_t s = 0; s < k; ++s)
        {
          for (size_t attempt = 0; attempt < maxAttempts; ++attempt)
          {
            size_t pidx = detail::sample_one_from_tree(
                tree, adaptor, vpt.data(), dist, radius_sq, engine, dim, subtreeSizes, min_correction);
            if (pidx != static_cast<size_t>(-1) && seen.find(pidx) == seen.end())
            {
              seen.insert(pidx);
              accepted.push_back(pidx);
              break;
            }
          }
        }
      }

      // Write accepted indices into the row
      for (size_t s = 0; s < accepted.size(); ++s)
      {
        allIndices({vantageIdx, s}) = accepted[s];
      }
      acceptedCounts[vantageIdx] = accepted.size();
    }, exec);

    // Check if all vantage points got exactly k samples
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
      return IndexedPointCloudCollection<T>(PointCloud<T>(source), std::move(allIndices));
    }

    // Some vantage points got fewer than k samples — compact the index tensor.
    // Find the max accepted count across all vantage points.
    size_t maxAccepted = *std::max_element(acceptedCounts.begin(), acceptedCounts.end());
    if (maxAccepted == 0)
    {
      // No samples accepted at all
      Tensor<size_t> emptyIndices({nVantage, 0_uz});
      return IndexedPointCloudCollection<T>(PointCloud<T>(source), std::move(emptyIndices));
    }

    Tensor<size_t> compactIndices({nVantage, maxAccepted});
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
          // Pad with index 0 (the point will be present but is a padding artifact)
          compactIndices({i, j}) = 0;
        }
      }
    }

    return IndexedPointCloudCollection<T>(PointCloud<T>(source), std::move(compactIndices));
  }

}

#endif
