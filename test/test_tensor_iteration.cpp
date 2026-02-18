/*
* Copyright 2024-2026 Bjorn Wehlin
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <gtest/gtest.h>

#include <mpcf/tensor.h>

static_assert(std::random_access_iterator<mpcf::Tensor1dValueIterator<mpcf::Tensor<double>>>);

TEST(TensorIteration, Iterate1dValues)
{
  mpcf::Tensor<int> x({3});
  x(0) = 1;
  x(1) = 2;
  x(2) = 3;

  auto begin = mpcf::begin1dValues(x);
  auto end = mpcf::end1dValues(x);

  EXPECT_EQ(*begin, 1);
  EXPECT_EQ(*(begin + 1), 2);
  EXPECT_EQ(*(begin + 2), 3);

  EXPECT_EQ(begin + 3, end);
}

TEST(TensorIteration, AxisIteration)
{
  mpcf::Tensor<int> x{{3, 3}};
  for (auto i = 0_uz; i < x.shape(0); ++i)
  {
    for (auto j = 0_uz; j < x.shape(1); ++j)
    {

    }
  }
  x({0, 0}) = 10;

}
