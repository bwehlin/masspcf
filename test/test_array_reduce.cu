#include <gtest/gtest.h>

#include <mpcf/array.h>
#include <mpcf/algorithms/array_reduce.h>


TEST(ArrayReduce, Shapes)
{
  mpcf::Array_f32 array({10, 5, 3});
  
  auto x = mpcf::array_reduce(array, 1, [](){});
  
  ASSERT_EQ(x.shape().size(), 2);
  EXPECT_EQ(x.shape()[0], 10);
  EXPECT_EQ(x.shape()[1], 3);
  
}
