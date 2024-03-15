#ifndef MPCF_ALGORITHM_ARRAY_REDUCE_H
#define MPCF_ALGORITHM_ARRAY_REDUCE_H

#include "../array.h"

namespace mpcf
{
  template <typename Tt, typename Tv, typename ReductionOp>
  Array<Tt, Tv> array_reduce(const Array<Tt, Tv>& inArr, size_t dim, ReductionOp op)
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
