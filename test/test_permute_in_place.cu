#include <gtest/gtest.h>

#include <mpcf/algorithms/permute_in_place.h>

TEST(PermuteInPlace, Permute1x1MatrixDoesNothing)
{
  std::vector<int> matrix{ 1 };
  std::vector<int> copy = matrix;
  std::vector<size_t> permutation{ 0 };

  mpcf::permute_in_place(matrix.data(), permutation);
  EXPECT_EQ(matrix, copy);
}
