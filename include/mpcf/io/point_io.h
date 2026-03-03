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

#ifndef MASSPCF_POINT_IO_H
#define MASSPCF_POINT_IO_H

#include "io_stream_base.h"
#include "../point.h"

namespace mpcf::io::detail
{
  template <PointLike PointT>
  void write_element(std::ostream& os, const PointT& pt)
  {
    write_bytes<typename PointT::time_type>(os, pt.t);
    write_bytes<typename PointT::value_type>(os, pt.v);
  }

  template <PointLike PointT>
  PointT read_element(std::istream& is)
  {
    PointT ret;
    ret.t = read_bytes<typename PointT::time_type>(is);
    ret.v = read_bytes<typename PointT::value_type>(is);
    return ret;
  }
}

#endif // MASSPCF_POINT_IO_H
