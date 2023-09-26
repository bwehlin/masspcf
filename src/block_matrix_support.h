#ifndef MPCF_BLOCK_MATRIX_SUPPORT_H
#define MPCF_BLOCK_MATRIX_SUPPORT_H

#include <cuda_runtime.h> // dim3
#include <string>

namespace mpcf::internal
{
  size_t get_row_size(size_t maxAllocationN, size_t nSplits, size_t nPcfs)
  {
    auto maxRowHeight = maxAllocationN / nPcfs;
    
    maxRowHeight = std::min(maxRowHeight, nPcfs);
    maxRowHeight /= nSplits;
    maxRowHeight = std::max(maxRowHeight, 1ul);
    
    return maxRowHeight;
  }
  
  std::vector<std::pair<size_t, size_t>> get_block_row_boundaries(size_t rowHeight, size_t nPcfs)
  {
    std::vector<std::pair<size_t, size_t>> boundaries;
    for (size_t i = 0ul;; i += rowHeight)
    {
      if (!boundaries.empty() && boundaries.back().second == nPcfs - 1)
      {
        return boundaries;
      }
      
      boundaries.emplace_back(i, std::min(i + rowHeight - 1ul, nPcfs));
      if (boundaries.back().second >= nPcfs)
      {
        boundaries.back().second = nPcfs - 1;
        return boundaries;
      }
    }
  }
  
  dim3 get_grid_dims(dim3 blockSz, size_t rowHeight, size_t nPcfs)
  {
    int xover = nPcfs % blockSz.x == 0 ? 0 : 1;
    int yover = rowHeight % blockSz.y == 0 ? 0 : 1;
    return dim3( nPcfs / blockSz.x + xover, rowHeight / blockSz.y + yover, 1 );
  }

}

inline bool operator==(const dim3& a, const dim3& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}



#endif
