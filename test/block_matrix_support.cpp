
#include <gtest/gtest.h>

#include "../src/block_matrix_support.h"

TEST(BlockMatrixSupport, Row1x1)
{
  auto boundaries = mpcf::internal::get_block_row_boundaries(1, 1);
  
  std::vector<std::pair<size_t, size_t>> expected(
    {std::make_pair(0ul, 0ul)});
  
  EXPECT_EQ(boundaries, expected);
}

TEST(BlockMatrixSupport, Row1x2)
{
  auto boundaries = mpcf::internal::get_block_row_boundaries(1, 2);
  
  std::vector<std::pair<size_t, size_t>> expected(
    {std::make_pair(0ul, 0ul), 
     std::make_pair(1ul, 1ul)});
  
  EXPECT_EQ(boundaries, expected);
}

TEST(BlockMatrixSupport, Row4x7)
{
  auto boundaries = mpcf::internal::get_block_row_boundaries(4, 7);
  
  std::vector<std::pair<size_t, size_t>> expected(
    {std::make_pair(0ul, 3ul), 
     std::make_pair(4ul, 6ul)});
  
  EXPECT_EQ(boundaries, expected);
  
}
