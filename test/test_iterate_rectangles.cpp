/*
* Copyright 2024 Bjorn Wehlin
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

#include <mpcf/pcf.h>
#include <mpcf/algorithms/matrix_integrate.h>
#include <mpcf/algorithms/iterate_rectangles.h>

#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include <type_traits>

class IterateRectanglesFixture : public ::testing::Test
{
protected:
  using Rectangle = typename mpcf::Pcf_f32::rectangle_type;
  using Point = typename mpcf::Pcf_f32::point_type;
  
  void SetUp() override
  {
    collectRectangle = [this](const Rectangle& rect){ rectangles.emplace_back(rect); };
    
    pcfs.emplace_back(mpcf::Pcf_f32{{0., 3.}, {1., 2.}, {4., 5.}, {6., 0.}});
    pcfs.emplace_back(mpcf::Pcf_f32{{0., 2.}, {3., 4.}, {4., 2.}, {5., 1.}, {8., 3.}});
  }
  
  std::vector<mpcf::Pcf_f32> pcfs;
  std::vector<Rectangle> rectangles;
  
  std::function<void(const Rectangle&)> collectRectangle;
};

TEST_F(IterateRectanglesFixture, Full)
{
  mpcf::iterate_rectangles(pcfs[0].points(), pcfs[1].points(), collectRectangle);
  
  ASSERT_EQ(rectangles.size(), 7);
  
  EXPECT_EQ(rectangles[0], Rectangle().l(0.).r(1.).fv(3.).gv(2.));
  EXPECT_EQ(rectangles[1], Rectangle().l(1.).r(3.).fv(2.).gv(2.));
  EXPECT_EQ(rectangles[2], Rectangle().l(3.).r(4.).fv(2.).gv(4.));
  EXPECT_EQ(rectangles[3], Rectangle().l(4.).r(5.).fv(5.).gv(2.));
  EXPECT_EQ(rectangles[4], Rectangle().l(5.).r(6.).fv(5.).gv(1.));
  EXPECT_EQ(rectangles[5], Rectangle().l(6.).r(8.).fv(0.).gv(1.));
  EXPECT_EQ(rectangles[6], Rectangle().l(8.).r(Point::infinite_time()).fv(0.).gv(3.));
}

TEST_F(IterateRectanglesFixture, EndEarly)
{
  mpcf::iterate_rectangles(pcfs[0].points(), pcfs[1].points(), collectRectangle, 0, 4.5);
  
  ASSERT_EQ(rectangles.size(), 4);
  
  EXPECT_EQ(rectangles[0], Rectangle().l(0.).r(1.).fv(3.).gv(2.));
  EXPECT_EQ(rectangles[1], Rectangle().l(1.).r(3.).fv(2.).gv(2.));
  EXPECT_EQ(rectangles[2], Rectangle().l(3.).r(4.).fv(2.).gv(4.));
  EXPECT_EQ(rectangles[3], Rectangle().l(4.).r(4.5).fv(5.).gv(2.));
}

TEST_F(IterateRectanglesFixture, StartLate)
{
  mpcf::iterate_rectangles(pcfs[0].points(), pcfs[1].points(), collectRectangle, 2.);
  
  ASSERT_EQ(rectangles.size(), 6);
  
  EXPECT_EQ(rectangles[0], Rectangle().l(2.).r(3.).fv(2.).gv(2.));
  EXPECT_EQ(rectangles[1], Rectangle().l(3.).r(4.).fv(2.).gv(4.));
  EXPECT_EQ(rectangles[2], Rectangle().l(4.).r(5.).fv(5.).gv(2.));
  EXPECT_EQ(rectangles[3], Rectangle().l(5.).r(6.).fv(5.).gv(1.));
  EXPECT_EQ(rectangles[4], Rectangle().l(6.).r(8.).fv(0.).gv(1.));
  EXPECT_EQ(rectangles[5], Rectangle().l(8.).r(Point::infinite_time()).fv(0.).gv(3.));
}

TEST_F(IterateRectanglesFixture, StartAfterEverything)
{
  mpcf::iterate_rectangles(pcfs[0].points(), pcfs[1].points(), collectRectangle, 10.);
  
  ASSERT_EQ(rectangles.size(), 1);

  EXPECT_EQ(rectangles[0], Rectangle().l(10.).r(Point::infinite_time()).fv(0.).gv(3.));
}

auto xv(xt::xarray<mpcf::Pcf_f32>& arr)
{
  const xt::xstrided_slice_vector sv;
  return xt::strided_view(arr, sv); 
}


TEST_F(IterateRectanglesFixture, asdf)
{
  auto const & f = pcfs[0];
  auto const & g = pcfs[1];
  
  xt::xarray<mpcf::Pcf_f32> arr{
    {f     , g         , f + g},
    {g + g , f         , f + f},
    {g,      f + f + g , f * 2.0}
  };
  
  xt::xstrided_slice_vector sv = {xt::all(), 0};
  //using my_view_type = decltype(xt::strided_view(std::declval<xt::xarray<mpcf::Pcf_f32>&&>(), sv));
  using my_view_type = decltype(xv(std::declval<xt::xarray<mpcf::Pcf_f32>&>()));
  my_view_type view = xt::strided_view(arr, sv); 

  for (auto i = 0; i < view.shape(0); ++i)
  {
    view(i).debug_print();
  }
  
  auto v2 = xt::strided_view(view, {xt::range(1, 2)});
  
  EXPECT_EQ(view(0), f);
  EXPECT_EQ(view(1), g + g);
  EXPECT_EQ(view(2), g);
  
}
