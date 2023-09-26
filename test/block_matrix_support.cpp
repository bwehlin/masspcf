
#include <gtest/gtest.h>

#include "../src/block_matrix_support.h"

void PrintTo(const dim3& d, std::ostream* os)
{
  *os << "dim3(" << d.x << ", " << d.y << ", " << d.z << ")";
}

TEST(BlockMatrixSupport, GetBlockRowBoundaries_1x1)
{
  auto boundaries = mpcf::internal::get_block_row_boundaries(1, 1);
  
  std::vector<std::pair<size_t, size_t>> expected(
    {std::make_pair(0ul, 0ul)});
  
  EXPECT_EQ(boundaries, expected);
}

TEST(BlockMatrixSupport, GetBlockRowBoundaries_1x2)
{
  auto boundaries = mpcf::internal::get_block_row_boundaries(1, 2);
  
  std::vector<std::pair<size_t, size_t>> expected(
    {std::make_pair(0ul, 0ul), 
     std::make_pair(1ul, 1ul)});
  
  EXPECT_EQ(boundaries, expected);
}

TEST(BlockMatrixSupport, GetBlockRowBoundaries_4x7)
{
  auto boundaries = mpcf::internal::get_block_row_boundaries(4, 7);
  
  std::vector<std::pair<size_t, size_t>> expected(
    {std::make_pair(0ul, 3ul), 
     std::make_pair(4ul, 6ul)});
  
  EXPECT_EQ(boundaries, expected);
}

TEST(BlockMatrixSupport, GetGridDims_1_111_1)
{
  auto rowHeight = 1ul;
  dim3 blockSz(1, 1, 1);
  int nPcfs = 1;
  
  auto dims = mpcf::internal::get_grid_dims(blockSz, rowHeight, nPcfs);
  EXPECT_EQ(dims, dim3(1,1,1));
}

TEST(BlockMatrixSupport, GetGridDims_overflows_1x1)
{
  auto rowHeight = 1ul;
  int nPcfs = 1;
  
  {
    dim3 blockSz(8, 1, 1);

    auto dims = mpcf::internal::get_grid_dims(blockSz, rowHeight, nPcfs);
    EXPECT_EQ(dims, dim3(1,1,1));
  }
  
  {
    dim3 blockSz(1, 8, 1);

    auto dims = mpcf::internal::get_grid_dims(blockSz, rowHeight, nPcfs);
    EXPECT_EQ(dims, dim3(1,1,1));
  }
  
  {
    dim3 blockSz(8, 8, 1);

    auto dims = mpcf::internal::get_grid_dims(blockSz, rowHeight, nPcfs);
    EXPECT_EQ(dims, dim3(1,1,1));
  }
}

TEST(BlockMatrixSupport, GetGridDims_overflows_1x9)
{
  auto rowHeight = 1ul;
  int nPcfs = 9;
  
  {
    dim3 blockSz(8, 1, 1);

    auto dims = mpcf::internal::get_grid_dims(blockSz, rowHeight, nPcfs);
    EXPECT_EQ(dims, dim3(2,1,1));
  }
  
  {
    dim3 blockSz(1, 8, 1);

    auto dims = mpcf::internal::get_grid_dims(blockSz, rowHeight, nPcfs);
    EXPECT_EQ(dims, dim3(9,1,1));
  }
  
  {
    dim3 blockSz(8, 8, 1);

    auto dims = mpcf::internal::get_grid_dims(blockSz, rowHeight, nPcfs);
    EXPECT_EQ(dims, dim3(2,1,1));
  }
}

TEST(BlockMatrixSupport, GetGridDims_exact_1x16)
{
  auto rowHeight = 1ul;
  int nPcfs = 16;
  
  {
    dim3 blockSz(8, 1, 1);

    auto dims = mpcf::internal::get_grid_dims(blockSz, rowHeight, nPcfs);
    EXPECT_EQ(dims, dim3(2,1,1));
  }
  
  {
    dim3 blockSz(1, 8, 1);

    auto dims = mpcf::internal::get_grid_dims(blockSz, rowHeight, nPcfs);
    EXPECT_EQ(dims, dim3(16,1,1));
  }
  
  {
    dim3 blockSz(8, 8, 1);

    auto dims = mpcf::internal::get_grid_dims(blockSz, rowHeight, nPcfs);
    EXPECT_EQ(dims, dim3(2,1,1));
  }
}

TEST(BlockMatrixSupport, GetGridDims_exact_32x16)
{
  auto rowHeight = 32ul;
  int nPcfs = 16;
  
  {
    dim3 blockSz(8, 1, 1);

    auto dims = mpcf::internal::get_grid_dims(blockSz, rowHeight, nPcfs);
    EXPECT_EQ(dims, dim3(2,32,1));
  }
  
  {
    dim3 blockSz(1, 8, 1);

    auto dims = mpcf::internal::get_grid_dims(blockSz, rowHeight, nPcfs);
    EXPECT_EQ(dims, dim3(16,4,1));
  }
  
  {
    dim3 blockSz(8, 8, 1);

    auto dims = mpcf::internal::get_grid_dims(blockSz, rowHeight, nPcfs);
    EXPECT_EQ(dims, dim3(2,4,1));
  }
}


TEST(BlockMatrixSupport, GetGridDims_overflow_33x17)
{
  auto rowHeight = 33ul;
  int nPcfs = 17;
  
  {
    dim3 blockSz(8, 1, 1);

    auto dims = mpcf::internal::get_grid_dims(blockSz, rowHeight, nPcfs);
    EXPECT_EQ(dims, dim3(3,33,1));
  }
  
  {
    dim3 blockSz(1, 8, 1);

    auto dims = mpcf::internal::get_grid_dims(blockSz, rowHeight, nPcfs);
    EXPECT_EQ(dims, dim3(17,5,1));
  }
  
  {
    dim3 blockSz(8, 8, 1);

    auto dims = mpcf::internal::get_grid_dims(blockSz, rowHeight, nPcfs);
    EXPECT_EQ(dims, dim3(3,5,1));
  }
}

