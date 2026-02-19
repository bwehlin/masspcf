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

#include "io_stream.h"

#include <iostream>
#include <string>
#include <string_view>
#include <stdexcept>

namespace
{
  constexpr const std::string_view versionId = "1.0";
}

namespace mpcf
{
  bool IStream::read_header()
  {
    char buf[256];

    ensured_read(buf, 1);

    return true;
  }

  void IStream::ensured_read(char *out, size_t n)
  {
    m_is.read(out, n);
    if (m_is.gcount() != n)
    {
      throw std::runtime_error("Invalid file");
    }
  }

  void OStream::write_header()
  {

    m_os << '\1' << "MPCF" << versionId << '\2';
  }
}
