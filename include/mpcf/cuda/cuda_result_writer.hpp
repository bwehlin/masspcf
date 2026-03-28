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

#ifndef MPCF_CUDA_RESULT_WRITER_HPP
#define MPCF_CUDA_RESULT_WRITER_HPP

#include "../distance_matrix.hpp"
#include "../symmetric_matrix.hpp"
#include "../tensor.hpp"
#include "cuda_block_scheduler.hpp"

#include <cstddef>

namespace mpcf
{
  /// Writes block results into a DistanceMatrix.
  /// Holds a copy of the DistanceMatrix (which shares data via shared_ptr),
  /// so it is safe to outlive the original.
  template <typename Tv>
  class DistanceMatrixResultWriter
  {
  public:
    explicit DistanceMatrixResultWriter(DistanceMatrix<Tv> distmat)
      : m_distmat(std::move(distmat))
    { }

    void scatter(const Tv* hostBlock, const BlockInfo& block) const
    {
      for (size_t iLocal = 0; iLocal < block.rowHeight; ++iLocal)
      {
        size_t iGlobal = block.rowStart + iLocal;
        for (size_t jLocal = 0; jLocal < block.colWidth; ++jLocal)
        {
          size_t jGlobal = block.colStart + jLocal;
          if (iGlobal <= jGlobal)
          {
            continue;
          }
          m_distmat(iGlobal, jGlobal) = hostBlock[iLocal * block.colWidth + jLocal];
        }
      }
    }

  private:
    mutable DistanceMatrix<Tv> m_distmat;
  };

  /// Writes block results into a SymmetricMatrix.
  /// Holds a copy of the SymmetricMatrix (which shares data via shared_ptr),
  /// so it is safe to outlive the original.
  template <typename Tv>
  class SymmetricMatrixResultWriter
  {
  public:
    explicit SymmetricMatrixResultWriter(SymmetricMatrix<Tv> symmat)
      : m_symmat(std::move(symmat))
    { }

    void scatter(const Tv* hostBlock, const BlockInfo& block) const
    {
      for (size_t iLocal = 0; iLocal < block.rowHeight; ++iLocal)
      {
        size_t iGlobal = block.rowStart + iLocal;
        for (size_t jLocal = 0; jLocal < block.colWidth; ++jLocal)
        {
          size_t jGlobal = block.colStart + jLocal;
          if (iGlobal < jGlobal)
          {
            continue;
          }
          m_symmat(iGlobal, jGlobal) = hostBlock[iLocal * block.colWidth + jLocal];
        }
      }
    }

  private:
    mutable SymmetricMatrix<Tv> m_symmat;
  };

  /// Writes block results into a dense Tensor (row-major).
  /// Holds a copy of the Tensor (which shares data via shared_ptr).
  template <typename Tv>
  class DenseResultWriter
  {
  public:
    DenseResultWriter(Tensor<Tv> out, size_t nCols)
      : m_out(std::move(out))
      , m_nCols(nCols)
    { }

    void scatter(const Tv* hostBlock, const BlockInfo& block) const
    {
      Tv* data = m_out.data();
      for (size_t iLocal = 0; iLocal < block.rowHeight; ++iLocal)
      {
        size_t iGlobal = block.rowStart + iLocal;
        for (size_t jLocal = 0; jLocal < block.colWidth; ++jLocal)
        {
          size_t jGlobal = block.colStart + jLocal;
          data[iGlobal * m_nCols + jGlobal] = hostBlock[iLocal * block.colWidth + jLocal];
        }
      }
    }

  private:
    mutable Tensor<Tv> m_out;
    size_t m_nCols;
  };
}

#endif
