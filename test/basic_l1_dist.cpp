#include <gtest/gtest.h>

#include <mpcf/pcf.h>
#include <mpcf/algorithms/cuda_matrix_integrate.h>
#include <mpcf/algorithms/matrix_integrate.h>

#include <vector>
#include <memory>

namespace
{
  class PcfL1Integrator : public ::testing::Test
  {
  public:
    std::vector<mpcf::Pcf_f32> pcfs;
    std::vector<float> matrix;
    void allocate_matrix()
    {
      matrix.resize(pcfs.size() * pcfs.size());
    }

    float ij(size_t i, size_t j) const
    {
      return matrix.at(i * pcfs.size() + j);
    }
    
  };
}

TEST_F(PcfL1Integrator, EmptyPcfPairL1dist)
{
  pcfs.resize(2);
  allocate_matrix();
  
  mpcf::matrix_l1_dist<float,float>(matrix.data(), pcfs);
  
  EXPECT_EQ(ij(0, 0), 0.f);
  EXPECT_EQ(ij(0, 1), 0.f);
  EXPECT_EQ(ij(1, 0), 0.f);
  EXPECT_EQ(ij(1, 1), 0.f);
}

TEST_F(PcfL1Integrator, TwoPointPcfL1dist)
{
  
  pcfs.emplace_back(std::vector<mpcf::Point_f32>({ mpcf::Point_f32(0.f, 3.f), mpcf::Point_f32(1.0, 0.0) }));
  pcfs.emplace_back(std::vector<mpcf::Point_f32>({ mpcf::Point_f32(0.f, 1.f), mpcf::Point_f32(2.0, 0.0) }));
  allocate_matrix();
  
  mpcf::matrix_l1_dist<float,float>(matrix.data(), pcfs);
  
  EXPECT_EQ(ij(0, 0), 0.f);
  EXPECT_EQ(ij(0, 1), 3.f);
  EXPECT_EQ(ij(1, 1), 0.f);
}
