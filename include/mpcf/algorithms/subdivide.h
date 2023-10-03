#ifndef MPCF_ALGORITHMS_SUBDIVIDE_H
#define MPCF_ALGORITHMS_SUBDIVIDE_H

#include <vector>
#include <utility>

namespace mpcf
{
  inline std::vector<std::pair<size_t, size_t>> 
  subdivide(size_t blockSize, size_t nItems)
  {
    std::vector<std::pair<size_t, size_t>> boundaries;
    for (size_t i = 0ul;; i += blockSize)
    {
      if (!boundaries.empty() && boundaries.back().second == nItems - 1)
      {
        return boundaries;
      }
      
      boundaries.emplace_back(i, std::min(i + blockSize - 1ul, nItems));
      if (boundaries.back().second >= nItems)
      {
        boundaries.back().second = nItems - 1;
        return boundaries;
      }
    }
  }
}

#endif
