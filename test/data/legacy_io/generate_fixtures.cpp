// Generates the legacy-format io fixtures in this directory by writing through
// an OLDER stablebear's io code (pre-indexed-views `main`), so the read tests
// in test_io_readwrite.cpp / test_io.py prove the current reader still accepts
// files written by released versions. On that `main`, PointCloud<T> is an
// alias for Tensor<T>, so point-cloud tensors are nested tensors written as
// baseFormat 1000 and distance-matrix tensors as baseFormat 1120 (the current
// writer uses the shared-source layouts 1001/1121).
//
// The checked-in fixtures were produced like this (devcontainer, repo at
// /workspaces/stablebear); regeneration should never be needed unless the
// fixture *data* below is deliberately changed — the whole point of the files
// is that their bytes stay frozen:
//
//   git -C /workspaces/stablebear archive main include | tar -x -C /tmp/legacy_gen
//   g++ -std=c++20 -I /tmp/legacy_gen/include -I /workspaces/stablebear/3rd/taskflow \
//       generate_fixtures.cpp -o /tmp/legacy_gen/gen
//   /tmp/legacy_gen/gen /workspaces/stablebear/test/data/legacy_io
//
// The deterministic values below are mirrored exactly by the read tests; keep
// the two in sync.

#include <sbear/io.hpp>
#include <sbear/tensor.hpp>

#include <fstream>
#include <iostream>
#include <string>

namespace sb
{
  // version.hpp only declares these; the definitions normally come from the
  // CMake-generated version.cpp. Fixed strings keep the fixture bytes
  // deterministic ("0.4.4" was main's version when the fixtures were written).
  extern const std::string PROJECT_NAME = "stablebear";
  extern const std::string PROJECT_TITLE = "stablebear";
  extern const std::string PROJECT_VERSION = "0.4.4";
  extern const std::string PROJECT_VERSION_FULL = "0.4.4";
  extern const std::string PROJECT_BUILD_DATE = "2026-01-01";
}

namespace
{

  // Cloud c: (3 - c) points, dim 2, coord (i, j) = 100c + 10i + j + 0.25.
  template <typename T>
  sb::Tensor<sb::Tensor<T>> make_pcloud_tensor()
  {
    sb::Tensor<sb::Tensor<T>> t({ 2 });
    for (size_t c = 0; c < 2; ++c)
    {
      sb::Tensor<T> cloud({ 3 - c, 2 });
      for (size_t i = 0; i < 3 - c; ++i)
      {
        for (size_t j = 0; j < 2; ++j)
        {
          cloud({ i, j }) = static_cast<T>(100 * c + 10 * i + j) + T(0.25);
        }
      }
      t({ c }) = cloud;
    }
    return t;
  }

  // Entry (i, j), i < j: 10i + j + offset.
  template <typename T>
  sb::DistanceMatrix<T> make_distmat(size_t n, T offset)
  {
    sb::DistanceMatrix<T> m(n);
    for (size_t i = 0; i < n; ++i)
    {
      for (size_t j = i + 1; j < n; ++j)
      {
        m(i, j) = static_cast<T>(10 * i + j) + offset;
      }
    }
    return m;
  }

  // Matrix c: size 3 + c, entries offset by 0.5.
  template <typename T>
  sb::Tensor<sb::DistanceMatrix<T>> make_distmat_tensor()
  {
    sb::Tensor<sb::DistanceMatrix<T>> t({ 2 });
    for (size_t c = 0; c < 2; ++c)
    {
      t({ c }) = make_distmat<T>(3 + c, T(0.5));
    }
    return t;
  }

  template <typename WriteFn>
  void write_file(const std::string& dir, const std::string& name, WriteFn&& writeFn)
  {
    const auto path = dir + "/" + name;
    std::ofstream os(path, std::ios::binary);
    if (!os)
    {
      throw std::runtime_error("cannot open " + path);
    }
    writeFn(os);
    std::cout << "wrote " << path << '\n';
  }

}

int main(int argc, char** argv)
{
  const std::string dir = argc > 1 ? argv[1] : ".";

  write_file(dir, "pcloud_tensor_f32.sb",
             [](std::ostream& os) { sb::write(make_pcloud_tensor<sb::float32_t>(), os); });
  write_file(dir, "pcloud_tensor_f64.sb",
             [](std::ostream& os) { sb::write(make_pcloud_tensor<sb::float64_t>(), os); });

  write_file(dir, "distmat_tensor_f32.sb",
             [](std::ostream& os) { sb::write(make_distmat_tensor<sb::float32_t>(), os); });
  write_file(dir, "distmat_tensor_f64.sb",
             [](std::ostream& os) { sb::write(make_distmat_tensor<sb::float64_t>(), os); });

  // Standalone (SingleObject) distance matrices, size 5, offset 0.25.
  write_file(dir, "distmat_object_f32.sb",
             [](std::ostream& os) { sb::write_object(make_distmat<sb::float32_t>(5, 0.25f), os); });
  write_file(dir, "distmat_object_f64.sb",
             [](std::ostream& os) { sb::write_object(make_distmat<sb::float64_t>(5, 0.25), os); });

  return 0;
}
