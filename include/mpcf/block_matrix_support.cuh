#ifndef MPCF_BLOCK_MATRIX_SUPPORT_CUH
#define MPCF_BLOCK_MATRIX_SUPPORT_CUH

#include <cuda_runtime.h> // dim3
#include <string>
#include <algorithm>

#include <mpcf/algorithms/subdivide.h>

namespace mpcf::internal
{
  inline size_t get_row_size(size_t maxAllocationN, size_t nSplits, size_t nPcfs)
  {
    size_t maxRowHeight = maxAllocationN / nPcfs;
    
    maxRowHeight = std::min(maxRowHeight, nPcfs);
    maxRowHeight /= nSplits;
    maxRowHeight = std::max<size_t>(maxRowHeight, 1ul);
    
    return maxRowHeight;
  }
  
  inline dim3 get_grid_dims(dim3 blockSz, size_t rowHeight, size_t nPcfs)
  {
    int iover = rowHeight % blockSz.x == 0 ? 0 : 1;
    int jover = nPcfs % blockSz.y == 0 ? 0 : 1;
    
    return dim3( rowHeight / blockSz.x + iover, nPcfs / blockSz.y + jover, 1 );
  }
  
  inline size_t get_row_height_from_boundaries(std::pair<size_t, size_t> boundaries)
  {
    return boundaries.second - boundaries.first + 1;
  }

}

inline bool operator==(const dim3& a, const dim3& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}



#endif
