#ifndef MPCF_PYPCF_SUPPORT_H
#define MPCF_PYPCF_SUPPORT_H

#include <mpcf/pcf.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace mpcf
{
  namespace detail
  {

    template <typename Tt, typename Tv>
    mpcf::Pcf<Tt, Tv> construct_pcf(py::array_t<Tt> arr)
    {
      using point_type = typename mpcf::Pcf<Tt, Tv>::point_type;

      std::vector<mpcf::Point<Tt, Tv>> points;

      py::buffer_info buf = arr.request();
      if (buf.size == 0)
      {
        points.emplace_back(static_cast<Tt>(0), static_cast<Tv>(0));
        return mpcf::Pcf<Tt, Tv>(std::move(points));
      }

      if (buf.ndim != 2)
      {
        throw std::runtime_error("Input array should have two dimensions (time + value).");
      }

      auto data = arr.template unchecked<2>();
      auto start = 0ul;
      if (data(0, 0) != 0)
      {
        ++start;
      }

      if (buf.shape[0] == 2)
      {
        points.resize(buf.shape[1] + start);
        points[0].t = 0;
        points[0].v = 0;
        for (auto i = 0; i < buf.shape[1]; ++i)
        {
          points[i + start].t = data(0, i);
          points[i + start].v = data(1, i);
        }
      }
      else if (buf.shape[1] == 2)
      {
        points.resize(buf.shape[0] + start);
        points[0].t = 0;
        points[0].v = 0;
        for (auto i = 0; i < buf.shape[0]; ++i)
        {
          points[i + start].t = data(i, 0);
          points[i + start].v = data(i, 1);
        }
      }
      else
      {
        throw std::runtime_error("Input array should be either 2xn or nx2 (it is " + std::to_string(buf.shape[0]) + "x" + std::to_string(buf.shape[1]) + ")");
      }

      auto sortByTime = [](const point_type& a, const point_type & b){
        return a.t < b.t;
      };

      if (!std::is_sorted(points.begin(), points.end(), sortByTime))
      {
        std::sort(points.begin(), points.end(), sortByTime);
      }

      return mpcf::Pcf<Tt, Tv>(std::move(points));
    }

    template <typename TPcf>
    py::memoryview to_numpy(const TPcf& pcf)
    {
      using TTime = typename TPcf::time_type;
      using TVal = typename TPcf::value_type;
      static_assert(std::is_same<TTime, TVal>::value, "time and value type must be the same");

      return py::memoryview::from_buffer(
        reinterpret_cast<const TTime*>(pcf.points().data()), 
        { Py_ssize_t(2), Py_ssize_t(pcf.points().size())},
        { Py_ssize_t(sizeof(TTime)), Py_ssize_t(sizeof(TTime) * 2)});
    }


  }
}

#endif
