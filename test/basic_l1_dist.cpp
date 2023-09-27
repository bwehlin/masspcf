#include <gtest/gtest.h>

#include <mpcf/pcf.h>
#include <mpcf/algorithms/cuda_matrix_integrate.h>
#include <mpcf/algorithms/matrix_integrate.h>

#include <vector>
#include <memory>

namespace
{
  class PcfL1IntegratorFixture : public ::testing::TestWithParam<mpcf::Executor>
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

TEST_P(PcfL1IntegratorFixture, EmptyPcfPairL1dist)
{
  auto executor = GetParam();
  
  pcfs.resize(2);
  allocate_matrix();
  
  mpcf::matrix_l1_dist<float,float>(matrix.data(), pcfs, executor);
  
  EXPECT_EQ(ij(0, 0), 0.f);
  EXPECT_EQ(ij(0, 1), 0.f);
  EXPECT_EQ(ij(1, 0), 0.f);
  EXPECT_EQ(ij(1, 1), 0.f);
}

TEST_P(PcfL1IntegratorFixture, TwoPointPcfL1dist)
{
  auto executor = GetParam();
  
  pcfs.emplace_back(std::vector<mpcf::Point_f32>({ mpcf::Point_f32(0.f, 3.f), mpcf::Point_f32(1.0, 0.0) }));
  pcfs.emplace_back(std::vector<mpcf::Point_f32>({ mpcf::Point_f32(0.f, 1.f), mpcf::Point_f32(2.0, 0.0) }));
  allocate_matrix();
  
  mpcf::matrix_l1_dist<float,float>(matrix.data(), pcfs, executor);
  
  EXPECT_EQ(ij(0, 0), 0.f);
  EXPECT_EQ(ij(0, 1), 3.f);
  EXPECT_EQ(ij(1, 1), 0.f);
}

INSTANTIATE_TEST_CASE_P(
    PcfL1Integrator,
    PcfL1IntegratorFixture,
    ::testing::Values(mpcf::Executor::Cpu, mpcf::Executor::Cuda),
    [](const testing::TestParamInfo<mpcf::Executor>& info) {
      switch (info.param)
      {
      case mpcf::Executor::Cpu: return "CPU";
      case mpcf::Executor::Cuda: return "CUDA";
      }
      return "<<UNKNOWN EXECUTOR>>";
    }
);
