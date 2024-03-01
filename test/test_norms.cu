#include <gtest/gtest.h>

#include <mpcf/pcf.h>
#include <mpcf/algorithms/apply_functional.h>

#include <vector>
#include <memory>

namespace
{
  TEST(L1Norm, EmptyPcfHasZeroNorm)
  {
    mpcf::Pcf_f32 f;
    EXPECT_EQ(mpcf::l1_norm(f), 0.f);
  }

  TEST(L1Norm, TwoPointPcf)
  {
    mpcf::Pcf_f32 f({ {0.f, 2.f}, {1.5f, 0.f} });

    EXPECT_FLOAT_EQ(mpcf::l1_norm(f), 3.f);
  }

  TEST(L1Norm, FourPointPcf)
  {
    mpcf::Pcf_f32 f({ {0.f, 2.f}, {1.5f, 1.f}, {3.f, 7.f}, {4.f, 0.f} });

    EXPECT_FLOAT_EQ(mpcf::l1_norm(f), 11.5f);
  }
}
