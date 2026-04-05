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

#ifndef MPCF_SAMPLING_INDEXED_POINT_CLOUD_H
#define MPCF_SAMPLING_INDEXED_POINT_CLOUD_H

#include "../tensor.hpp"

#include <memory>
#include <stdexcept>

namespace mpcf::sampling
{

  /// A lightweight indexed view into a source PointCloud.
  /// Shares data (ref-counted) with the source and selects rows via an index array.
  template <ArithmeticType T>
  class IndexedPointCloud
  {
  public:
    IndexedPointCloud(std::shared_ptr<const PointCloud<T>> source, Tensor<size_t> indices)
      : m_source(std::move(source))
      , m_indices(std::move(indices))
    {
      if (m_source->rank() != 2)
      {
        throw std::invalid_argument("source must have rank 2 (N x D)");
      }
      if (m_indices.rank() != 1)
      {
        throw std::invalid_argument("indices must have rank 1");
      }
    }

    size_t n_points() const { return m_indices.shape()[0]; }
    size_t dim() const { return m_source->shape()[1]; }

    T operator()(size_t point, size_t d) const
    {
      return (*m_source)({m_indices({point}), d});
    }

    const Tensor<size_t>& indices() const { return m_indices; }
    const PointCloud<T>& source() const { return *m_source; }

    /// Materialize into a contiguous PointCloud (copies data).
    PointCloud<T> materialize() const
    {
      size_t n = n_points();
      size_t D = dim();
      PointCloud<T> result({n, D});
      for (size_t i = 0; i < n; ++i)
      {
        size_t src_idx = m_indices({i});
        for (size_t d = 0; d < D; ++d)
        {
          result({i, d}) = (*m_source)({src_idx, d});
        }
      }
      return result;
    }

  private:
    std::shared_ptr<const PointCloud<T>> m_source;
    Tensor<size_t> m_indices;
  };

  /// Groups one source cloud with multiple index arrays — the natural output of the sampler.
  /// Stores a shared source of shape (N, D) and an index tensor of shape (M, k).
  template <ArithmeticType T>
  class IndexedPointCloudCollection
  {
  public:
    IndexedPointCloudCollection(PointCloud<T> source, Tensor<size_t> indices)
      : m_source(std::make_shared<PointCloud<T>>(std::move(source)))
      , m_indices(std::move(indices))
    {
      if (m_source->rank() != 2)
      {
        throw std::invalid_argument("source must have rank 2 (N x D)");
      }
      if (m_indices.rank() != 2)
      {
        throw std::invalid_argument("indices must have rank 2 (M x k)");
      }
    }

    size_t n_vantage() const { return m_indices.shape()[0]; }
    size_t k() const { return m_indices.shape()[1]; }
    size_t dim() const { return m_source->shape()[1]; }

    /// Get the i-th indexed view (indices row i).
    IndexedPointCloud<T> operator[](size_t i) const
    {
      if (i >= n_vantage())
      {
        throw std::out_of_range("index out of range");
      }

      size_t nk = k();
      Tensor<size_t> row_indices({nk});
      for (size_t j = 0; j < nk; ++j)
      {
        row_indices({j}) = m_indices({i, j});
      }
      return IndexedPointCloud<T>(m_source, std::move(row_indices));
    }

    const PointCloud<T>& source() const { return *m_source; }
    const Tensor<size_t>& indices() const { return m_indices; }

  private:
    std::shared_ptr<PointCloud<T>> m_source;
    Tensor<size_t> m_indices;  // shape (M, k)
  };

}

#endif
