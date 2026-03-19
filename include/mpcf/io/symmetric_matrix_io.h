// Copyright 2024-2026 Bjorn Wehlin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MASSPCF_SYMMETRIC_MATRIX_IO_H
#define MASSPCF_SYMMETRIC_MATRIX_IO_H

#include "io_stream_base.h"
#include "../symmetric_matrix.h"

namespace mpcf::io::detail
{
  template <typename T>
  void write_element(std::ostream& os, const SymmetricMatrix<T>& mat)
  {
    write_bytes<uint64_t>(os, mat.n());
    for (size_t i = 0; i < mat.storage_count(); ++i)
    {
      write_bytes<T>(os, mat.data()[i]);
    }
  }

  template <typename T>
  SymmetricMatrix<T> read_symmetric_matrix(std::istream& is)
  {
    auto n = read_bytes<uint64_t>(is);
    SymmetricMatrix<T> mat(n);
    for (size_t i = 0; i < mat.storage_count(); ++i)
    {
      mat.data()[i] = read_bytes<T>(is);
    }
    return mat;
  }
}

#endif // MASSPCF_SYMMETRIC_MATRIX_IO_H
