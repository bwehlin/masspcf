#include "algorithms/cuda_matrix_integrate.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

namespace
{
  template <typename T>
  __device__
  float l1_inner_prod(T, T, T t, T b)
  {
    return t * b;
  }
  
  __device__
  float l1_inner_prod_f32_impl(float l, float r, float t, float b)
  {
    return l1_inner_prod<float>(l, r, t, b);
  }
  
  __device__
  double l1_inner_prod_f64_impl(double l, double r, double t, double b)
  {
    return l1_inner_prod<double>(l, r, t, b);
  }
}

mpcf::DeviceOp<float, float>
mpcf::device_ops::l1_inner_prod_f32()
{
  return &l1_inner_prod_f32_impl;
}

mpcf::DeviceOp<double, double>
mpcf::device_ops::l1_inner_prod_f64()
{
  return &l1_inner_prod_f64_impl;
}

void
mpcf::detail::cuda_matrix_integrate_f32(float* out, const std::vector<Pcf_f32>& fs, DeviceOp<float, float> op)
{
  std::cout << "Hello" << std::endl;
}

void
mpcf::detail::cuda_matrix_integrate_f64(double* out, const std::vector<Pcf_f64>& fs, DeviceOp<double, double> op)
{
  std::cout << "Hello64" << std::endl;
}
