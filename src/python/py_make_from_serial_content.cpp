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

  namespace detail
  {
    using EnumerationDt = long long int;
  }

  template <typename Tt, typename Tv>
  NdArray<Tt, Tv>
    make_from_serial_content(py::array_t<Tt> content, py::array_t<detail::EnumerationDt> enumeration)
  {
    auto content_buf = content.request();
    auto enumeration_buf = enumeration.request();

    if (content_buf.ndim != 2)
    {
      throw std::runtime_error("content should have 2 dimensions");
    }

    if (enumeration_buf.ndim < 2)
    {
      throw std::runtime_error("enumeration must have at least 2 dimensions");
    }

    auto contentData = content.template unchecked<2>();
    auto enumerationData = enumeration.unchecked();

    using PointT = typename mpcf::Pcf<Tt, Tv>::point_type;

    auto nDataPoints = contentData.shape(0);

    auto nPcfs = enumerationData.shape(0);

    std::vector<size_t> targetShapeVec(enumeration_buf.ndim - 1);
    for (auto i = 0; i < enumeration_buf.ndim - 1; ++i) // Last dim is always 2 for [start, end)
    {
      targetShapeVec[i] = enumerationData.shape(i);
    }
    Shape targetShape(std::move(targetShapeVec));

    NdArray<Tt, Tv> pcfs = NdArray<Tt, Tv>::make_zeros(targetShape);

    auto sourceData = static_cast<Tv*>(enumeration_buf.ptr);
    auto sourceStrides = enumeration_buf.strides;

    auto targetStrides = pcfs.data().strides();

    if (targetStrides.size() + 1 != sourceStrides.size())
    {
      throw std::runtime_error("incompatible strides");
    }

    for (auto i = 0; i < targetStrides.size(); ++i)
    {
      if (enumerationData.shape(i) != 1)
      {
        // xtensor treats the stride associated with dimension 1 on an axis as 0 (we can't move to the second element
        // along that axis, so might as well leave stride at 0), whereas array_t has a "proper" stride. We therefore
        // skip checking this case.
        auto expected = targetStrides[i]
          * sizeof(detail::EnumerationDt) // array_t strides are in bytes, xtensor strides are in elements
          * 2; // enumeration contains [start, end], so each PCF has 2 entries
        auto actual = sourceStrides[i];

        if (actual != expected)
        {
          throw std::runtime_error("unexpected stride in dimension " + std::to_string(i) + " (expected " + std::to_string(expected) + " but got " + std::to_string(actual) + ")");
        }
      }
    }

    // Flatten should be safe as we checked strides earlier
    auto targetFlatView = xt::flatten(pcfs.data());
    auto* enumerationPtr = static_cast<const detail::EnumerationDt*>(enumeration_buf.ptr);

    for (ssize_t ei = 0; ei < targetFlatView.size(); ++ei)
    {
      auto start = enumerationPtr[ei];
      auto end = enumerationPtr[ei + 1];

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
      if (points.empty())
      {
        points.emplace_back(0, 0);
      }

      pcfs.data()[ei] = mpcf::Pcf<Tt, Tv>(std::move(points));
    }

    return pcfs;
  }

  void register_make_from_serial_content(py::module_& m)
  {
    m.def("make_from_serial_content_32", &make_from_serial_content<float, float>);
    m.def("make_from_serial_content_64", &make_from_serial_content<double, double>);
  }

}