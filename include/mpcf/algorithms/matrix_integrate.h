#ifndef MPCF_ALGORITHMS_MATRIX_INTEGRATE_H
#define MPCF_ALGORITHMS_MATRIX_INTEGRATE_H

#include "iterate_rectangles.h"
#include "../pcf.h"
#include "../executor.h"

#ifdef BUILD_WITH_CUDA
#include "cuda_matrix_integrate.h"
#endif

#include <vector>

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
  
  template <typename Tt, typename Tv, typename RectangleOp>
  void matrix_integrate(Executor& exec, Tv* out, const std::vector<Pcf<Tt, Tv>>& fs, const RectangleOp& op, bool symmetric = false, Tt a = 0.f, Tt b = std::numeric_limits<Tt>::max())
  {
    auto sz = fs.size();
    for (auto i = 0ul; i < sz; ++i)
    {
      auto j = symmetric ? i : 0ul; // For now, fill entire matrix
      for (; j < sz; ++j)
      {
        out[i * sz + j] = integrate<Tt, Tv, RectangleOp>(fs[i], fs[j], op, a, b);
      }
    }
    
    if (symmetric)
    {
      // Build lower triangle
      size_t sz = fs.size();
      for (size_t i = 1; i < sz; ++i)
      {
        for (size_t j = 0; j < i; ++j)
        {
          out[i * sz + j] = out[j * sz + i];
        }
      }
    }
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
      matrix_integrate(out, fs, [](const rect_t& rect) -> Tv {
        return std::abs(rect.top - rect.bottom);
      }, true);
    }
  }
}

#endif
