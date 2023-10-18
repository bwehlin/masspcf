#ifndef MPCF_ALGORITHMS_MATRIX_INTEGRATE_H
#define MPCF_ALGORITHMS_MATRIX_INTEGRATE_H

#include "iterate_rectangles.h"
#include "../pcf.h"
#include "../executor.h"

#ifdef BUILD_WITH_CUDA
#include "cuda_matrix_integrate.h"
#endif

#include <vector>

#include <taskflow/algorithm/for_each.hpp>

namespace mpcf
{
  template <typename Tt, typename Tv, typename RectangleOp>
  Tv integrate(const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g, RectangleOp op, Tt a = 0.f, Tt b = std::numeric_limits<Tt>::max())
  {
    using rect_t = typename Pcf<Tt, Tv>::rectangle_type;

    Tv val = 0.f;
    iterate_rectangles(f, g, a, b, [&val, &op](const rect_t& rect) -> void {
      val += (rect.right - rect.left) * op(rect);
    });
    
    return val;
  }
  
  template <typename Tv>
  void make_lower_triangle(Executor& exec, Tv* out, size_t sz)
  {
    tf::Taskflow flow;
    flow.for_each_index(0ul, sz, 1ul, [out, sz](size_t i) {
      for (size_t j = 0; j < i; ++j)
      {
        out[i * sz + j] = out[j * sz + i];
      }
    });
    auto future = exec->run(flow);
    future.wait();
  }
  
  template <typename Tt, typename Tv, typename RectangleOp>
  void matrix_integrate(Executor& exec, Tv* out, const std::vector<Pcf<Tt, Tv>>& fs, const RectangleOp& op, bool symmetric = true, Tt a = 0.f, Tt b = std::numeric_limits<Tt>::max())
  {
    auto sz = fs.size();
    tf::Taskflow flow;
    
    flow.for_each_index(0ul, sz, 1ul, [out, &fs, &op, symmetric, a, b, sz](size_t i) {
      auto j = symmetric ? i : 0ul;
      for (; j < sz; ++j)
      {
        out[i * sz + j] = integrate<Tt, Tv, RectangleOp>(fs[i], fs[j], op, a, b);
      }
    });

    auto future = exec->run(flow);
    future.wait();
  }
  
  template <typename Tt, typename Tv, typename RectangleOp>
  void matrix_integrate(Tv* out, const std::vector<Pcf<Tt, Tv>>& fs, const RectangleOp& op, bool symmetric = false, Tt a = 0.f, Tt b = std::numeric_limits<Tt>::max())
  {
    matrix_integrate<Tt, Tv, RectangleOp>(default_cpu_executor(), out, fs, op, symmetric, a, b);
  }
  
  template <typename Tt, typename Tv>
  void matrix_l1_dist(Tv* out, const std::vector<Pcf<Tt, Tv>>& fs, Executor& executor = default_cpu_executor())
  {
    using rect_t = typename Pcf<Tt, Tv>::rectangle_type;
    
    switch (executor.hardware())
    {
#ifdef BUILD_WITH_CUDA
    case Hardware::CUDA:
      cuda_matrix_l1_dist<Tt, Tv>(out, fs);
      break;
#endif
    default:
      matrix_integrate(executor, out, fs, [](const rect_t& rect) -> Tv {
        return std::abs(rect.top - rect.bottom);
      }, true);
    }
    
    auto & cpuExec = executor.hardware() == Hardware::CPU ? executor : default_cpu_executor();
    make_lower_triangle<Tv>(cpuExec, out, fs.size());
  }
}

#endif
