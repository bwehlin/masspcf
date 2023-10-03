#ifndef MPCF_BLOCK_MATRIX_SUPPORT_H
#define MPCF_BLOCK_MATRIX_SUPPORT_H

#include <cuda_runtime.h> // dim3
#include <string>

#include <mpcf/algorithms/subdivide.h>

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
  
  dim3 get_grid_dims(dim3 blockSz, size_t rowHeight, size_t nPcfs)
  {
    int iover = rowHeight % blockSz.x == 0 ? 0 : 1;
    int jover = nPcfs % blockSz.y == 0 ? 0 : 1;
    
    return dim3( rowHeight / blockSz.x + iover, nPcfs / blockSz.y + jover, 1 );
  }
  
  size_t get_row_height_from_boundaries(std::pair<size_t, size_t> boundaries)
  {
    return boundaries.second - boundaries.first + 1;
  }

}

inline bool operator==(const dim3& a, const dim3& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}



#endif
