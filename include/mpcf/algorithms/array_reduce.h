#ifndef MPCF_ALGORITHM_ARRAY_REDUCE_H
#define MPCF_ALGORITHM_ARRAY_REDUCE_H

#include "../array.h"
#include "../executor.h"
#include "reduce.h"

namespace mpcf
{
  bool next_array_index(std::vector<size_t>& pos, const std::vector<size_t>& shape)
  {
    if (shape.empty() || shape.size() != pos.size())
    {
      return false;
    }

    auto sz = shape.size();
    auto m = sz - 1;
    
    if (pos[0] == 1 && pos[1] == 2)
    {
      int a = 0;
      ++a;
    }
    
    // Find first zero
    auto iStart = size_t(0);
    for (auto iStart = 0; 
         pos[iStart] != 0 && iStart < m; 
         ++iStart)
    { }
    
    for (auto i = iStart; i < m; ++i)
    {
      if (pos[i] + 1 < shape[i])
      {
        ++pos[i];
        return true;
      }
      
      // Increment 'one up' and reset all lower
      ++pos[i + 1];
      for (auto j = 0; j < i + 1; ++j)
      {
        pos[j] = 0;
      }
      
      return true;
    }
    
    // Nothing more to do
    return false;
  }
  
  template <typename Tt, typename Tv, typename ReductionOp>
  Array<Tt, Tv> array_reduce(const Array<Tt, Tv>& inArr, size_t dim, ReductionOp op, Executor& exec = default_executor())
  {
    if (inArr.shape().empty())
    {
      return inArr;
    }
    
    std::vector<size_t> newShape = inArr.shape();
    newShape.erase(newShape.begin() + dim);
    Array<Tt, Tv> outArr(newShape);
    
    
    
    return outArr;
  }
}

#endif
