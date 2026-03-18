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

// --- walk tests ---

TEST(TensorWalk, Walk1dOdometerOrder)
{
  mpcf::Tensor<int> x({4});
  std::vector<std::vector<size_t>> visited;
  mpcf::walk(x, [&](const std::vector<size_t>& idx) {
    visited.push_back(idx);
  });

  std::vector<std::vector<size_t>> expected = {{0}, {1}, {2}, {3}};
  EXPECT_EQ(visited, expected);
}

TEST(TensorWalk, Walk2dOdometerOrder)
{
  mpcf::Tensor<int> x({2, 3});
  std::vector<std::vector<size_t>> visited;
  mpcf::walk(x, [&](const std::vector<size_t>& idx) {
    visited.push_back(idx);
  });

  std::vector<std::vector<size_t>> expected = {
    {0, 0}, {0, 1}, {0, 2},
    {1, 0}, {1, 1}, {1, 2}
  };
  EXPECT_EQ(visited, expected);
}

TEST(TensorWalk, Walk3dOdometerOrder)
{
  mpcf::Tensor<int> x({2, 2, 2});
  std::vector<std::vector<size_t>> visited;
  mpcf::walk(x, [&](const std::vector<size_t>& idx) {
    visited.push_back(idx);
  });

  std::vector<std::vector<size_t>> expected = {
    {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
    {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}
  };
  EXPECT_EQ(visited, expected);
}

TEST(TensorWalk, WalkEmptyTensor)
{
  mpcf::Tensor<int> x({0});
  size_t count = 0;
  mpcf::walk(x, [&](const std::vector<size_t>&) { ++count; });
  EXPECT_EQ(count, 0);
}

TEST(TensorWalk, WalkEmptyShape)
{
  mpcf::Tensor<int> x(std::vector<size_t>{});
  size_t count = 0;
  mpcf::walk(x, [&](const std::vector<size_t>&) { ++count; });
  EXPECT_EQ(count, 0);
}

TEST(TensorWalk, WalkBoolEarlyTermination)
{
  mpcf::Tensor<int> x({10});
  std::vector<std::vector<size_t>> visited;
  mpcf::walk(x, [&](const std::vector<size_t>& idx) -> bool {
    visited.push_back(idx);
    return idx[0] < 3;
  });

  std::vector<std::vector<size_t>> expected = {{0}, {1}, {2}, {3}};
  EXPECT_EQ(visited, expected);
}

TEST(TensorWalk, WalkReadsCorrectValues)
{
  mpcf::Tensor<int> x({2, 3});
  int val = 0;
  x({0, 0}) = val++;
  x({0, 1}) = val++;
  x({0, 2}) = val++;
  x({1, 0}) = val++;
  x({1, 1}) = val++;
  x({1, 2}) = val++;

  std::vector<int> values;
  mpcf::walk(x, [&](const std::vector<size_t>& idx) {
    values.push_back(x(idx));
  });

  std::vector<int> expected = {0, 1, 2, 3, 4, 5};
  EXPECT_EQ(values, expected);
}

TEST(TensorWalk, WalkSingleElement)
{
  mpcf::Tensor<int> x({1});
  std::vector<std::vector<size_t>> visited;
  mpcf::walk(x, [&](const std::vector<size_t>& idx) {
    visited.push_back(idx);
  });

  std::vector<std::vector<size_t>> expected = {{0}};
  EXPECT_EQ(visited, expected);
}

TEST(TensorWalk, WalkZeroDimInMiddle)
{
  mpcf::Tensor<int> x({3, 0, 2});
  size_t count = 0;
  mpcf::walk(x, [&](const std::vector<size_t>&) { ++count; });
  EXPECT_EQ(count, 0);
}

TEST(TensorWalk, MemberWalkMatchesFreeWalk)
{
  mpcf::Tensor<int> x({2, 3});

  std::vector<std::vector<size_t>> from_member;
  x.walk([&](const std::vector<size_t>& idx) {
    from_member.push_back(idx);
  });

  std::vector<std::vector<size_t>> from_free;
  mpcf::walk(x, [&](const std::vector<size_t>& idx) {
    from_free.push_back(idx);
  });

  EXPECT_EQ(from_member, from_free);
}
