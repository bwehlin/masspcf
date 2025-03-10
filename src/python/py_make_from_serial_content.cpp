/*
* Copyright 2024-2025 Bjorn Wehlin
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "py_make_from_serial_content.h"
#include "pyarray.h"
#include <iostream>

namespace py = pybind11;

namespace mpcf_py
{

  template <typename Tt, typename Tv>
  NdArray<Tt, Tv>
    make_from_serial_content(py::array_t<Tt> content, py::array_t<long> enumeration)
  {
    std::cout << "Hello1\n" << std::flush;
    auto content_buf = content.request();
    auto enumeration_buf = content.request();

    if (content_buf.ndim != 2)
    {
      throw std::runtime_error("content should have 2 dimensions");
    }

    if (enumeration_buf.ndim != 2)
    {
      throw std::runtime_error("enumeration should have 2 dimensions");
    }

    auto contentData = content.template unchecked<2>();
    auto enumerationData = enumeration.template unchecked<2>();

    using PointT = typename mpcf::Pcf<Tt, Tv>::point_type;

    auto nDataPoints = contentData.shape(0);
    auto nPcfs = enumerationData.shape(0);


    NdArray<Tt, Tv> pcfs = NdArray<Tt, Tv>::make_zeros({ size_t(nPcfs) });

    for (ssize_t ei = 0; ei < nPcfs; ++ei)
    {
      auto start = enumerationData(ei, 0);
      auto end = enumerationData(ei, 1);

      if (end < start)
      {
        throw std::runtime_error("end < start at enumeration index " + std::to_string(ei));
      }

      if (end > nDataPoints)
      {
        throw std::runtime_error("content has fewer points than enumeration endpoint " + std::to_string(ei));
      }

      std::vector<PointT> points;
      points.reserve(end - start);
      for (auto pi = start; pi < end; ++pi)
      {
        points.emplace_back(contentData(pi, 0), contentData(pi, 1));
      }

      pcfs.at({size_t(ei)}) = mpcf::Pcf<Tt, Tv>(std::move(points));
    }

    return pcfs;
  }

  void register_make_from_serial_content(py::module_& m)
  {
    m.def("make_from_serial_content_32", &make_from_serial_content<float, float>);
    m.def("make_from_serial_content_64", &make_from_serial_content<double, double>);
  }

}