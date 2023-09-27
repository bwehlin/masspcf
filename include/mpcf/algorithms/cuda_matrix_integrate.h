#ifndef MPCF_ALGORITHMS_CUDA_MATRIX_COMBINE_H
#define MPCF_ALGORITHMS_CUDA_MATRIX_COMBINE_H

#include "../pcf.h"
//#include "reduce.h"

#include <functional>
#include <vector>

namespace mpcf
{
  template <typename Tt, typename Tv>
  using DeviceOp = Tv (*)(Tt, Tt, Tv, Tv);
  
  namespace detail
  {
    void cuda_matrix_integrate_f32(float* out, const std::vector<Pcf_f32>& fs, DeviceOp<float, float> op);
    void cuda_matrix_integrate_f64(double* out, const std::vector<Pcf_f64>& fs, DeviceOp<double, double> op);
  }
  
  template <typename Tt, typename Tv>
  void cuda_matrix_integrate(Tv* out, const std::vector<Pcf<Tt, Tv>>& fs, DeviceOp<Tt, Tv> op)
  {
    if constexpr (std::is_same<Tt, float>::value && std::is_same<Tv, float>::value)
    {
      detail::cuda_matrix_integrate_f32(out, fs, op);
    }
    else if constexpr (std::is_same<Tt, double>::value && std::is_same<Tv, double>::value)
    {
      detail::cuda_matrix_integrate_f64(out, fs, op);
    }
#if 0
    else
    {
      static_assert(false, "Unknown PCF type");
    }
#endif
  }
  
  namespace device_ops
  {
    DeviceOp<float, float> l1_inner_prod_f32();
    DeviceOp<double, double> l1_inner_prod_f64();
    
    template <typename Tt, typename Tv>
    DeviceOp<Tt, Tv> l1_inner_prod()
    {
      if constexpr (std::is_same<Tt, float>::value && std::is_same<Tv, float>::value)
      {
        return l1_inner_prod_f32();
      }
      else if constexpr (std::is_same<Tt, double>::value && std::is_same<Tv, double>::value)
      {
        return l1_inner_prod_f64();
      }
      // TODO fail
    }
  }
  
}

#endif
