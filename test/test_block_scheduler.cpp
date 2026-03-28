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

#include <mpcf/cuda/cuda_block_scheduler.hpp>

#include <set>
#include <utility>

TEST(CudaBlockScheduler, EmptyMatrix)
{
  mpcf::CudaBlockScheduler scheduler({
    .nRows = 0, .nCols = 0,
    .maxOutputElements = 1000,
    .nSplitsHint = 1,
    .triangleMode = mpcf::BlockTriangleMode::LowerTriangle
  });

  EXPECT_TRUE(scheduler.blocks().empty());
}

TEST(CudaBlockScheduler, SingleElement)
{
  mpcf::CudaBlockScheduler scheduler({
    .nRows = 1, .nCols = 1,
    .maxOutputElements = 1000,
    .nSplitsHint = 1,
    .triangleMode = mpcf::BlockTriangleMode::LowerTriangle
  });

  auto const& blocks = scheduler.blocks();
  ASSERT_EQ(blocks.size(), 1);
  EXPECT_EQ(blocks[0].rowStart, 0);
  EXPECT_EQ(blocks[0].rowHeight, 1);
  EXPECT_EQ(blocks[0].colStart, 0);
  EXPECT_EQ(blocks[0].colWidth, 1);
}

TEST(CudaBlockScheduler, SmallMatrix_LowerTriangle)
{
  mpcf::CudaBlockScheduler scheduler({
    .nRows = 4, .nCols = 4,
    .maxOutputElements = 4,
    .nSplitsHint = 1,
    .triangleMode = mpcf::BlockTriangleMode::LowerTriangle
  });

  auto const& blocks = scheduler.blocks();

  // Verify all lower-triangle (i,j) pairs with i >= j are covered exactly once
  std::set<std::pair<size_t, size_t>> covered;
  for (auto const& block : blocks)
  {
    for (size_t i = block.rowStart; i < block.rowStart + block.rowHeight; ++i)
    {
      for (size_t j = block.colStart; j < block.colStart + block.colWidth; ++j)
      {
        if (i >= j)
        {
          auto [it, inserted] = covered.insert({i, j});
          EXPECT_TRUE(inserted) << "Duplicate coverage at (" << i << ", " << j << ")";
        }
      }
    }
  }

  // Check all lower-triangle pairs are present
  for (size_t i = 0; i < 4; ++i)
  {
    for (size_t j = 0; j <= i; ++j)
    {
      EXPECT_TRUE(covered.count({i, j})) << "Missing coverage at (" << i << ", " << j << ")";
    }
  }
}

TEST(CudaBlockScheduler, SmallMatrix_Full)
{
  mpcf::CudaBlockScheduler scheduler({
    .nRows = 4, .nCols = 4,
    .maxOutputElements = 4,
    .nSplitsHint = 1,
    .triangleMode = mpcf::BlockTriangleMode::Full
  });

  auto const& blocks = scheduler.blocks();

  // Verify all (i,j) pairs are covered exactly once
  std::set<std::pair<size_t, size_t>> covered;
  for (auto const& block : blocks)
  {
    for (size_t i = block.rowStart; i < block.rowStart + block.rowHeight; ++i)
    {
      for (size_t j = block.colStart; j < block.colStart + block.colWidth; ++j)
      {
        auto [it, inserted] = covered.insert({i, j});
        EXPECT_TRUE(inserted) << "Duplicate coverage at (" << i << ", " << j << ")";
      }
    }
  }

  for (size_t i = 0; i < 4; ++i)
  {
    for (size_t j = 0; j < 4; ++j)
    {
      EXPECT_TRUE(covered.count({i, j})) << "Missing coverage at (" << i << ", " << j << ")";
    }
  }
}

TEST(CudaBlockScheduler, LowerTriangleHasFewerBlocks)
{
  mpcf::CudaBlockScheduler lower({
    .nRows = 10, .nCols = 10,
    .maxOutputElements = 25,
    .nSplitsHint = 1,
    .triangleMode = mpcf::BlockTriangleMode::LowerTriangle
  });

  mpcf::CudaBlockScheduler full({
    .nRows = 10, .nCols = 10,
    .maxOutputElements = 25,
    .nSplitsHint = 1,
    .triangleMode = mpcf::BlockTriangleMode::Full
  });

  EXPECT_LT(lower.blocks().size(), full.blocks().size());
}

TEST(CudaBlockScheduler, LargeMatrixSmallBlocks)
{
  mpcf::CudaBlockScheduler scheduler({
    .nRows = 100, .nCols = 100,
    .maxOutputElements = 100,
    .nSplitsHint = 4,
    .triangleMode = mpcf::BlockTriangleMode::LowerTriangle
  });

  auto const& blocks = scheduler.blocks();
  EXPECT_GT(blocks.size(), 1);

  // Verify no block exceeds budget
  for (auto const& block : blocks)
  {
    EXPECT_LE(block.rowHeight * block.colWidth, 100)
      << "Block (" << block.rowStart << "," << block.colStart
      << ") size " << block.rowHeight << "x" << block.colWidth << " exceeds budget";
  }

  // Verify all lower-triangle pairs covered
  std::set<std::pair<size_t, size_t>> covered;
  for (auto const& block : blocks)
  {
    for (size_t i = block.rowStart; i < block.rowStart + block.rowHeight; ++i)
    {
      for (size_t j = block.colStart; j < block.colStart + block.colWidth; ++j)
      {
        if (i >= j)
        {
          covered.insert({i, j});
        }
      }
    }
  }

  for (size_t i = 0; i < 100; ++i)
  {
    for (size_t j = 0; j <= i; ++j)
    {
      EXPECT_TRUE(covered.count({i, j})) << "Missing coverage at (" << i << ", " << j << ")";
    }
  }
}

TEST(CudaBlockScheduler, MaxRowHeightAndColWidth)
{
  mpcf::CudaBlockScheduler scheduler({
    .nRows = 10, .nCols = 10,
    .maxOutputElements = 10000,
    .nSplitsHint = 1,
    .triangleMode = mpcf::BlockTriangleMode::Full
  });

  // With budget=10000 and nPcfs=10, one block should cover everything
  EXPECT_EQ(scheduler.max_row_height(), 10);
  EXPECT_EQ(scheduler.max_col_width(), 10);
  EXPECT_EQ(scheduler.blocks().size(), 1);
}

TEST(CudaBlockScheduler, SortedByDescendingWork)
{
  mpcf::CudaBlockScheduler scheduler({
    .nRows = 20, .nCols = 20,
    .maxOutputElements = 100,
    .nSplitsHint = 1,
    .triangleMode = mpcf::BlockTriangleMode::LowerTriangle
  });

  auto const& blocks = scheduler.blocks();
  for (size_t i = 1; i < blocks.size(); ++i)
  {
    EXPECT_GE(blocks[i - 1].rowHeight * blocks[i - 1].colWidth,
              blocks[i].rowHeight * blocks[i].colWidth);
  }
}

TEST(CudaBlockScheduler, RectangularMatrix_Full)
{
  mpcf::CudaBlockScheduler scheduler({
    .nRows = 3,
    .nCols = 5,
    .maxOutputElements = 100,
    .nSplitsHint = 1,
    .triangleMode = mpcf::BlockTriangleMode::Full
  });

  auto const& blocks = scheduler.blocks();

  // Verify all (i,j) pairs in a 3x5 matrix are covered
  std::set<std::pair<size_t, size_t>> covered;
  for (auto const& block : blocks)
  {
    for (size_t i = block.rowStart; i < block.rowStart + block.rowHeight; ++i)
    {
      for (size_t j = block.colStart; j < block.colStart + block.colWidth; ++j)
      {
        auto [it, inserted] = covered.insert({i, j});
        EXPECT_TRUE(inserted) << "Duplicate coverage at (" << i << ", " << j << ")";
      }
    }
  }

  for (size_t i = 0; i < 3; ++i)
  {
    for (size_t j = 0; j < 5; ++j)
    {
      EXPECT_TRUE(covered.count({i, j})) << "Missing coverage at (" << i << ", " << j << ")";
    }
  }
}

TEST(CudaBlockScheduler, RectangularMatrix_ManyBlocks)
{
  mpcf::CudaBlockScheduler scheduler({
    .nRows = 7,
    .nCols = 13,
    .maxOutputElements = 10,
    .nSplitsHint = 1,
    .triangleMode = mpcf::BlockTriangleMode::Full
  });

  auto const& blocks = scheduler.blocks();
  EXPECT_GT(blocks.size(), 1);

  // Verify complete coverage
  std::set<std::pair<size_t, size_t>> covered;
  for (auto const& block : blocks)
  {
    for (size_t i = block.rowStart; i < block.rowStart + block.rowHeight; ++i)
    {
      for (size_t j = block.colStart; j < block.colStart + block.colWidth; ++j)
      {
        covered.insert({i, j});
      }
    }
  }

  EXPECT_EQ(covered.size(), 7 * 13);
}
