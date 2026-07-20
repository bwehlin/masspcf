#ifndef SB_ALGORITHMS_MINIMUM_SPANNING_TREE_H
#define SB_ALGORITHMS_MINIMUM_SPANNING_TREE_H

#include "../tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace sb
{
  /// Squared-Euclidean distance functor over a point cloud: answers d(i, j)^2
  /// on demand without materializing a distance matrix. Exposes the same call
  /// interface as DistanceMatrix<T> (operator()(i, j) and size()), so
  /// algorithms such as MST construction can be templated over either. Works
  /// in squared distances because comparisons are unchanged (x -> x^2 is
  /// monotone on [0, inf)); the caller applies sqrt to the few distances it
  /// keeps (e.g. the n-1 merge distances of an MST) instead of one sqrt per
  /// O(n^2) query.
  template <typename T>
  class SquaredEuclideanDistance
  {
  public:
    explicit SquaredEuclideanDistance(const PointCloud<T> &points)
        : m_data(points.data()), m_pointStride(points.stride(0)), m_coordStride(points.stride(1)),
          m_size(points.shape(0)), m_dim(points.shape(1))
    {
    }

    [[nodiscard]] T operator()(size_t i, size_t j) const
    {
      const T *p = m_data + static_cast<ptrdiff_t>(i) * m_pointStride;
      const T *q = m_data + static_cast<ptrdiff_t>(j) * m_pointStride;
      T sumSq{0};
      for (auto k = ptrdiff_t{0}; k < static_cast<ptrdiff_t>(m_dim); ++k)
      {
        auto diff = p[k * m_coordStride] - q[k * m_coordStride];
        sumSq += diff * diff;
      }
      return sumSq;
    }

    [[nodiscard]] size_t size() const noexcept
    {
      return m_size;
    }

  private:
    const T *m_data;
    ptrdiff_t m_pointStride;
    ptrdiff_t m_coordStride;
    size_t m_size;
    size_t m_dim;
  };

  template <typename T>
  struct MergeEdge
  {
    size_t a; // endpoints of the MST edge = representatives of the two merging components
    size_t b;
    T mergeDist;
  };

  // Distance type is anything answering d(i, j) on demand: in practice a
  // distance matrix, or an oracle such as SquaredEuclideanDistance that
  // computes distances from a point cloud without materializing a matrix.
  //
  // Fills merges with the n-1 single-linkage merges of the metric, sorted ascending
  // by merge distance. Array-based Prim: O(n^2) distance queries, O(n) memory.
  template <typename DistT, typename T>
  void mst_merge_order(const DistT &dist, size_t n, std::vector<MergeEdge<T>> &merges)
  {
    merges.clear();
    if (n <= 1)
    {
      return;
    }
    merges.reserve(n - 1);

    std::vector<T> minDist(n, std::numeric_limits<T>::infinity());
    std::vector<size_t> parent(n, 0);
    std::vector<char> visited(n, 0);

    minDist[0] = T{0};

    for (size_t iter = 0; iter < n; ++iter)
    {
      // Pick the unvisited node closest to the tree.
      size_t nearest = 0;
      bool found = false;
      for (size_t j = 0; j < n; ++j)
      {
        if (!visited[j] && (!found || minDist[j] < minDist[nearest]))
        {
          nearest = j;
          found = true;
        }
      }

      visited[nearest] = 1;
      if (iter > 0)
      {
        merges.push_back({parent[nearest], nearest, minDist[nearest]});
      }

      for (size_t j = 0; j < n; ++j)
      {
        if (!visited[j])
        {
          auto candidateDist = dist(nearest, j);
          if (candidateDist < minDist[j])
          {
            minDist[j] = candidateDist;
            parent[j] = nearest;
          }
        }
      }
    }

    std::sort(merges.begin(), merges.end(), [](const MergeEdge<T> &lhs, const MergeEdge<T> &rhs) {
      return lhs.mergeDist < rhs.mergeDist;
    });
  }
} // namespace sb

#endif
