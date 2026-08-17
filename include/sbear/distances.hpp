#ifndef STABLEBEAR_DISTANCES_H
#define STABLEBEAR_DISTANCES_H

#include "concepts.hpp"
#include "tensor.hpp"

#include <cstddef>

namespace sb
{
  // Distance oracles: lightweight functors that answer d(i, j) on demand over
  // some underlying data, satisfying the DistanceOracle concept (see
  // concepts.hpp). Together with DistanceMatrix<T> they let algorithms be
  // templated over "anything that measures distance between indexed points" --
  // wrapping data such as a point cloud in an oracle is the intended way to
  // add new metrics without materializing matrices.

  /// Squared-Euclidean distance functor over a point cloud: answers d(i, j)^2
  /// on demand without materializing a distance matrix. Satisfies
  /// DistanceOracle just like DistanceMatrix<T>, so algorithms such as MST
  /// construction can be templated over either.
  /// Precondition: the cloud must be rank 2 with shape (n, dim) — the
  /// constructor reads stride(1)/shape(1), so the caller must validate the
  /// rank first (as homological_kernel_pcloud_single_impl does). Works
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
} // namespace sb

#endif
