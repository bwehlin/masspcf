/*
* Copyright 2024-2025 Bjorn Wehlin
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
    size_t iover = rowHeight % blockSz.x == 0 ? 0 : 1;
    size_t jover = nPcfs % blockSz.y == 0 ? 0 : 1;
    
    auto x = rowHeight / static_cast<size_t>(blockSz.x) + iover;
    auto y = nPcfs / static_cast<size_t>(blockSz.y) + jover;

    return dim3(static_cast<unsigned int>(x), static_cast<unsigned int>(y), 1);
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
