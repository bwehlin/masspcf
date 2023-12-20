#include <gtest/gtest.h>

#include <mpcf/algorithms/permute_in_place.cuh>

TEST(Permutations, EmptyPermutationHasNoCycles)
{
  std::vector<size_t> perm;
  auto cycles = mpcf::get_cycles(perm);
  EXPECT_TRUE(cycles.empty());
}

TEST(Permutations, SortedPermutationHasNoCycles)
{
  std::vector<size_t> perm{0,1,2,3};
  auto cycles = mpcf::get_cycles(perm);
  EXPECT_TRUE(cycles.empty());
}

TEST(Permutations, Cycle12)
{
  std::vector<size_t> perm{ 0,2,1,3 };
  auto cycles = mpcf::get_cycles(perm);
  ASSERT_EQ(cycles.size(), 1ul);
  EXPECT_EQ(cycles[0], std::vector<size_t>({ 1, 2 }));
}

TEST(Permutations, TwoCycles)
{
  std::vector<size_t> perm{ 0,2,1,4,6,5,3 };
  auto cycles = mpcf::get_cycles(perm);
  ASSERT_EQ(cycles.size(), 2ul);
  EXPECT_EQ(cycles[0], std::vector<size_t>({ 1, 2 }));
  EXPECT_EQ(cycles[1], std::vector<size_t>({ 3, 4, 6 }));
}

TEST(Permutations, TwoCyclesWithSpace)
{
  std::vector<size_t> perm{ 0,2,1,3,4,6,5,7 };
  auto cycles = mpcf::get_cycles(perm);
  ASSERT_EQ(cycles.size(), 2ul);
  EXPECT_EQ(cycles[0], std::vector<size_t>({ 1, 2 }));
  EXPECT_EQ(cycles[1], std::vector<size_t>({ 5, 6 }));
}

TEST(Permutations, ApplyPermutation)
{
  std::vector<std::string> vec{"a", "b", "c", "d"};
  std::vector<size_t> perm{0,2,1,3};
  auto cycles = mpcf::get_cycles(perm);
  mpcf::apply_permutation(vec.begin(), cycles);
  std::vector<std::string> expected{"a", "c", "b", "d"};
  EXPECT_EQ(vec, expected);
}

TEST(Permutations, ApplyPermutationLarger)
{
  std::vector<std::string> vec{"a", "b", "c", "d", "e", "f", "g", "h"};
  std::vector<size_t> perm{5,2,3,4,1,0,7,6};
  auto cycles = mpcf::get_cycles(perm);
  mpcf::invert_permutation(cycles);
  mpcf::apply_permutation(vec.begin(), cycles);
  std::vector<std::string> expected{"f", "c", "d", "e", "b", "a", "h", "g"};
  EXPECT_EQ(vec, expected);
}
