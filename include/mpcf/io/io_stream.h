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

#ifndef MASSPCF_IO_STREAM_H
#define MASSPCF_IO_STREAM_H

#include "../config.h"

#include <cstddef>
#include <iostream>

namespace mpcf::io::detail
{

  template <typename CharT, typename Traits>
  void assert_not_bad(std::basic_ios<CharT, Traits>& stream)
  {
    if (stream.bad())
    {
      throw std::runtime_error("Bad stream.");
    }
  }

  inline void write_binary_string(std::ostream& os, const std::string& str)
  {
    os.write(str.c_str(), str.size());
    assert_not_bad(os);
  }

  inline void write_binary_record(std::ostream& os, const std::string& str)
  {
    write_binary_string(os, str);
    write_binary_string(os, "\36"); // \36 is ASCII record separator (decimal 30)
  }

  template <typename DiskT, typename MemoryT = DiskT>
  requires std::is_convertible_v<DiskT, std::decay_t<MemoryT>>
  void write_bytes(std::ostream& os, MemoryT&& v)
  {
    auto && cv = static_cast<DiskT>(v);
    os.write(reinterpret_cast<const char*>(&cv), sizeof(cv));
    assert_not_bad(os);
  }

  template <typename DiskT, typename MemoryT = DiskT>
  requires std::is_convertible_v<DiskT, MemoryT>
  MemoryT read_bytes(std::istream& is)
  {
    DiskT v;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    assert_not_bad(is);
    if (is.gcount() != sizeof(v))
    {
      throw std::runtime_error("Incorrect number of bytes returned (file may be corrupted)");
    }
    return static_cast<MemoryT>(v);
  }

  template <typename T, std::forward_iterator FwdIt>
  requires std::is_convertible_v<typename FwdIt::value_type, T>
  void write_bytes(std::ostream& os, FwdIt begin, FwdIt end)
  {
    auto len = std::distance(begin, end);
    if (len < 0)
    {
      throw std::runtime_error("Negative length container (this is a bug, please report it!).");
    }
    write_bytes<mpcf::uint64_t>(os, len);
    for (auto it = begin; it != end; ++it)
    {
      write_bytes<T>(os, *it);
    }
  }

  inline void write_string(std::ostream& os, const std::string& str)
  {
    write_bytes<uint64_t>(os, str.length());
    os << str;
  }


}

#endif //MASSPCF_IO_STREAM_H
