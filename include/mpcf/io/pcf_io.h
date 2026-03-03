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

#ifndef MASSPCF_PCF_IO_H
#define MASSPCF_PCF_IO_H

#include "io_stream.h"
#include "../pcf.h"

namespace mpcf::io::detail
{
  template <typename Tt, typename Tv>
  void write_element(std::ostream& os, const Pcf<Tt, Tv>& pcf)
  {

  }
}

#endif // MASSPCF_PCF_IO_H
