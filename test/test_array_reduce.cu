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

TEST(ArrayReduce, NextIndex)
{
  std::vector<size_t> shape{2, 3, 2};
  std::vector<size_t> index{0, 0, 0};
  
  ASSERT_TRUE(mpcf::next_array_index(index, shape)) << "From 000";
  EXPECT_EQ(index, std::vector<size_t>({1, 0, 0})) << "From 000";
  
  ASSERT_TRUE(mpcf::next_array_index(index, shape)) << "From 100";
  EXPECT_EQ(index, std::vector<size_t>({0, 1, 0})) << "From 100";
  
  ASSERT_TRUE(mpcf::next_array_index(index, shape)) << "From 010";
  EXPECT_EQ(index, std::vector<size_t>({1, 1, 0})) << "From 010";
  
  ASSERT_TRUE(mpcf::next_array_index(index, shape)) << "From 110";
  EXPECT_EQ(index, std::vector<size_t>({0, 2, 0})) << "From 110";
  
  ASSERT_TRUE(mpcf::next_array_index(index, shape)) << "From 020";
  EXPECT_EQ(index, std::vector<size_t>({1, 2, 0})) << "From 020";
  
  ASSERT_TRUE(mpcf::next_array_index(index, shape)) << "From 120";
  EXPECT_EQ(index, std::vector<size_t>({0, 0, 1})) << "From 120";
  
  ASSERT_TRUE(mpcf::next_array_index(index, shape)) << "From 001";
  EXPECT_EQ(index, std::vector<size_t>({1, 0, 1})) << "From 001";
  
  ASSERT_TRUE(mpcf::next_array_index(index, shape)) << "From 101";
  EXPECT_EQ(index, std::vector<size_t>({0, 1, 1})) << "From 101";
  
  ASSERT_TRUE(mpcf::next_array_index(index, shape)) << "From 011";
  EXPECT_EQ(index, std::vector<size_t>({1, 1, 1})) << "From 011";
  
  ASSERT_TRUE(mpcf::next_array_index(index, shape)) << "From 111";
  EXPECT_EQ(index, std::vector<size_t>({0, 2, 1})) << "From 111";
  
  ASSERT_TRUE(mpcf::next_array_index(index, shape)) << "From 021";
  EXPECT_EQ(index, std::vector<size_t>({1, 2, 1})) << "From 021";
  
  ASSERT_FALSE(mpcf::next_array_index(index, shape)) << "From 121";
}
