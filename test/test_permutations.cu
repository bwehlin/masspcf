#include <gtest/gtest.h>

#include <mpcf/algorithms/permute_in_place.h>

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

