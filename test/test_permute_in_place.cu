#include <gtest/gtest.h>

#include <sstream>

#include <mpcf/algorithms/permute_in_place.cuh>

namespace
{
  std::vector<int> get_test_matrix(const std::vector<size_t>& permutation)
  {
    std::vector<int> ret;
    auto n = permutation.size();
    ret.resize(n * n);
    for (size_t i = 0ul; i < n; ++i)
    {
      auto sigma_i = permutation[i];
      for (size_t j = 0ul; j < n; ++j)
      {
        auto sigma_j = permutation[j];
        ret[i * n + j] = 10 * (sigma_i + 1) + (sigma_j + 1);
      }
    }
    return ret;
  }

  std::vector<int> get_test_matrix(size_t n)
  {
    std::vector<int> ret;
    ret.resize(n * n);
    for (size_t i = 0ul; i < n; ++i)
    {
      for (size_t j = 0ul; j < n; ++j)
      {
        ret[i * n + j] = 10 * (i + 1) + (j + 1);
      }
    }
    return ret;
  }

  std::string pretty_matrix(const std::vector<int>& matrix)
  {
    std::stringstream ss;
    size_t n = std::sqrt(matrix.size());
    for (auto i = 0ul; i < matrix.size(); ++i)
    {
      if (i != 0ul)
      {
        if (i % n == 0ul)
        {
          ss << '\n';
        }
        else
        {
          ss << ' ';
        }
      }
      ss << matrix[i];
    }
    return ss.str();
  }
}

TEST(PermuteInPlace, ReversePermute1x1MatrixDoesNothing)
{
  std::vector<int> matrix{ 1 };
  std::vector<int> copy = matrix;
  std::vector<size_t> permutation{ 0 };

  mpcf::reverse_permute_in_place(matrix.data(), permutation);
  EXPECT_EQ(matrix, copy);
}

TEST(PermuteInPlace, ReversePermute2x2MatrixWithFlip)
{
  std::vector<size_t> permutation{ 1, 0 };
  auto matrix = get_test_matrix(permutation);
  auto target = get_test_matrix(permutation.size());
    
  mpcf::reverse_permute_in_place(matrix.data(), permutation);
  EXPECT_EQ(matrix, target) << pretty_matrix(matrix) << "\n vs \n" << pretty_matrix(target);
}

TEST(PermuteInPlace, ReversePermute7x7MatrixWithTwoCycles)
{
  std::vector<size_t> permutation{ 0,2,1,4,6,5,3 };
  auto matrix = get_test_matrix(permutation);
  auto target = get_test_matrix(permutation.size());

  mpcf::reverse_permute_in_place(matrix.data(), permutation);
  EXPECT_EQ(matrix, target) << pretty_matrix(matrix) << "\n vs \n" << pretty_matrix(target);
}
