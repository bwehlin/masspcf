#ifndef MPCF_ALGORITHM_REDUCE_H
#define MPCF_ALGORITHM_REDUCE_H

#include "iterate_rectangles.h"

#include <functional>
#include <vector>
#include <numeric>
#include <algorithm>

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/reduce.hpp>


namespace mpcf
{
  template <typename TPcf>
  using TOp = std::function<typename TPcf::value_type(const typename TPcf::rectangle_type&)>;

  template <typename TPcf>
  TPcf combine(const TPcf& f, const TPcf& g, TOp<TPcf> op)
  {
    using point_type = typename TPcf::point_type;
    using rectangle_type = typename TPcf::rectangle_type;

    std::vector<point_type> retPts;
    std::size_t npts = 0;
    iterate_rectangles(f, g, 0, 1000, [&npts](const rectangle_type&){ ++npts; });
    retPts.resize(npts);
    auto i = 0ul;
    iterate_rectangles(f, g, 0, 1000, [&retPts, &i, &op](const rectangle_type& rect){
      retPts[i++] = point_type(rect.left, op(rect));
    });

    return TPcf(std::move(retPts));
  }

  template <typename TPcf>
  TPcf reduce(const std::vector<TPcf>& fs, TOp<TPcf> op)
  {
    return std::reduce(fs.begin(), fs.end(), TPcf(), [&op](const TPcf& f, const TPcf& g) {
      return combine(f, g, op);
    });
  }
  
  template <typename TPcf>
  TPcf parallel_reduce(const std::vector<TPcf>& fs, TOp<TPcf> op)
  {
    tf::Taskflow taskflow;
    tf::Executor exec;
    TPcf f;
    /*auto task =*/ taskflow.reduce(fs.begin(), fs.end(), f, [&op](const TPcf& f, const TPcf& g) {
      return combine(f, g, op);
    });
    exec.run(taskflow).wait();
    return f;
  }
  
  template <typename T>
  class VectorStorage
  {
  public:
    struct VectorEntry
    {
      std::vector<T> vec;
      bool isFree;
    };
    
  private:
    std::vector<
  };
  
  template <typename T>
  class free_tree_allocator : public std::allocator
  {
  public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
  };
  
  template <typename TPcf>
  TPcf parallel_mem_reduce(const std::vector<TPcf>& fs, TOp<TPcf> op)
  {
    
  }
}

#endif
