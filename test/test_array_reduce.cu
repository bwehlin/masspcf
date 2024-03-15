#include <gtest/gtest.h>

#include <mpcf/array.h>
#include <mpcf/algorithms/array_reduce.h>
#include <mpcf/reduce_ops.cuh>

TEST(ArrayReduce, Shapes)
{
  mpcf::Array_f32 array({10, 5, 3});
  
  auto x = mpcf::array_reduce(array, 1, [](){});
  
  ASSERT_EQ(x.shape().size(), 2);
  EXPECT_EQ(x.shape()[0], 10);
  EXPECT_EQ(x.shape()[1], 3);
}

TEST(ArrayReduce, Add3x2)
{
  mpcf::Array_f64 array({3, 2});
  
  mpcf::Pcf_f64 f00({{0., 1.}, {1., 3.}});
  mpcf::Pcf_f64 f01({{0., 2.}, {2., 1.}});
  
  mpcf::Pcf_f64 f10({{0., 1.}, {5., 6.}});
  mpcf::Pcf_f64 f11({{0., 10.}, {2., 2.}});
  
  mpcf::Pcf_f64 f20({{0., 3.}, {1., 2.}, {2., 0.}});
  mpcf::Pcf_f64 f21({{0., 1.}, {1., 3.}});
  
  array({0,0}) = f00;
  array({0,1}) = f01;
  array({1,0}) = f10;
  array({1,1}) = f11;
  array({2,0}) = f20;
  array({2,1}) = f21;
  
  auto x = mpcf::array_reduce(array, 0, std::plus<>());
  
  EXPECT_EQ(x({0}), f00 + f10 + f11);
}
