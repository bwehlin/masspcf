#ifndef MPCF_BLOCK_MATRIX_SUPPORT_H
#define MPCF_BLOCK_MATRIX_SUPPORT_H

namespace mpcf::internal
{
  size_t get_row_size(size_t maxAllocationN, size_t nSplits, size_t nPcfs)
  {
    auto maxRowSz = maxAllocationN / nPcfs;
    
    maxRowSz = std::min(maxRowSz, nPcfs);
    maxRowSz /= nSplits;
    maxRowSz = std::max(maxRowSz, 1ul);
    
    return maxRowSz;
  }
  
  std::vector<std::pair<size_t, size_t>> get_block_row_boundaries(size_t rowSz, size_t nPcfs)
  {
    std::vector<std::pair<size_t, size_t>> boundaries;
    for (size_t i = 0ul;; i += rowSz)
    {
      if (!boundaries.empty() && boundaries.back().second == nPcfs - 1)
      {
        return boundaries;
      }
      
      boundaries.emplace_back(i, std::min(i + rowSz - 1ul, nPcfs));
      if (boundaries.back().second >= nPcfs)
      {
        boundaries.back().second = nPcfs - 1;
        return boundaries;
      }
    }
  }
}

#endif
