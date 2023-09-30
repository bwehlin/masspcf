#ifndef MPCF_ALGORITHMS_MATRIX_INTEGRATE_H
#define MPCF_ALGORITHMS_MATRIX_INTEGRATE_H

#include "iterate_rectangles.h"
#include "../pcf.h"

#ifdef BUILD_WITH_CUDA
#include "cuda_matrix_integrate.h"
#endif

#include <vector>

namespace mpcf
{
  enum class Executor
  {
    Cpu,
    Cuda
  };
  
  template <typename Tt, typename Tv, typename RectangleOp>
  Tv integrate(const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g, RectangleOp op, Tt a = 0.f, Tt b = std::numeric_limits<Tt>::max(), bool print=false)
  {
    using rect_t = typename Pcf<Tt, Tv>::rectangle_type;

    Tv val = 0.f;
    iterate_rectangles(f, g, a, b, [&val, &op, print](const rect_t& rect) -> void {
      val += (rect.right - rect.left) * op(rect);
      if (print)
        std::cout << rect.right - rect.left << " " << op(rect) << '\n';
    });
    
    return val;
  }
  
  template <typename Tt, typename Tv, typename RectangleOp>
  void matrix_integrate(Tv* out, const std::vector<Pcf<Tt, Tv>>& fs, const RectangleOp& op, bool symmetric = false, Tt a = 0.f, Tt b = std::numeric_limits<Tt>::max())
  {
    auto sz = fs.size();
    for (auto i = 0ul; i < sz; ++i)
    {
      auto j = symmetric ? i : 0ul;
      for (; j < sz; ++j)
      {
        out[i * sz + j] = integrate<Tt, Tv, RectangleOp>(fs[i], fs[j], op, a, b);
      }
    }
  }
  
  template <typename Tt, typename Tv>
  void matrix_l1_dist(Tv* out, const std::vector<Pcf<Tt, Tv>>& fs, Executor executor = Executor::Cpu)
  {
    using rect_t = typename Pcf<Tt, Tv>::rectangle_type;
    
    switch (executor)
    {
#ifdef BUILD_WITH_CUDA
    case Executor::Cuda:
      cuda_matrix_l1_dist<Tt, Tv>(out, fs);
      break;
#endif
    default:
      matrix_integrate(out, fs, [](const rect_t& rect) -> Tv {
        return std::abs(rect.top - rect.bottom);
      }, true, Tt(0), Tt(6));
    }
  }
}

#endif
