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
}
