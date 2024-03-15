#include <gtest/gtest.h>

#include <mpcf/array.h>

TEST(Array, EmptyArrayHasNoStrides)
{
  mpcf::Array_f32 array;
  EXPECT_TRUE(array.strides().empty());
}

TEST(Array, NdArrayStrides)
{
  mpcf::Array_f32 array({10, 5, 3});
  ASSERT_EQ(array.strides().size(), size_t(3));
  EXPECT_EQ(array.strides()[0], size_t(15));
  EXPECT_EQ(array.strides()[1], size_t(3));
  EXPECT_EQ(array.strides()[2], size_t(1));
}

TEST(Array, NdArrayShape)
{
  mpcf::Array_f32 array({10, 5, 3});
  ASSERT_EQ(array.shape().size(), size_t(3));
  EXPECT_EQ(array.shape()[0], size_t(10));
  EXPECT_EQ(array.shape()[1], size_t(5));
  EXPECT_EQ(array.shape()[2], size_t(3));
}

TEST(Array, NdArrayLinearIndexing)
{
  mpcf::Array_f32 array({10, 5, 3});
  
  EXPECT_EQ(array.get_linear_index({0, 0, 0}), 0);
  EXPECT_EQ(array.get_linear_index({0, 0, 2}), 2);
  EXPECT_EQ(array.get_linear_index({0, 2, 0}), 2*3);
  EXPECT_EQ(array.get_linear_index({2, 0, 0}), 2*15);
  EXPECT_EQ(array.get_linear_index({4, 3, 2}), 4*15 + 3*3 + 2);
}
