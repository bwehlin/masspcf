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

#ifndef MASSPCF_TIMESERIES_IO_H
#define MASSPCF_TIMESERIES_IO_H

#include "io_stream_base.hpp"
#include "../timeseries.hpp"

namespace mpcf::io::detail
{
  template <typename T>
  void write_element(std::ostream& os, const T& ts)
    requires is_timeseries_v<T>
          && std::is_arithmetic_v<typename T::value_type>
  {
    using Tt = typename T::time_type;
    using Tv = typename T::value_type;

    if (ts.has_custom_strategy())
    {
      throw std::runtime_error(
          "TimeSeries with a custom interpolation strategy cannot be "
          "serialized; only the built-in InterpolationMode (nearest/linear) "
          "is persisted. Re-attach the strategy on load.");
    }

    write_bytes<uint64_t>(os, ts.n_times());
    write_bytes<uint64_t>(os, ts.n_channels());
    write_bytes<Tt>(os, ts.start_time());
    write_bytes<Tt>(os, ts.time_step());
    write_bytes<uint8_t>(os, static_cast<uint8_t>(ts.interpolation()));

    for (auto& t : ts.times())
      write_bytes<Tt>(os, t);

    for (auto& v : ts.values())
      write_bytes<Tv>(os, v);
  }

  template <typename TsT>
    requires std::is_arithmetic_v<typename TsT::value_type>
  TsT read_timeseries(std::istream& is)
  {
    using Tt = typename TsT::time_type;
    using Tv = typename TsT::value_type;

    auto n_times = read_bytes<uint64_t>(is);
    auto n_channels = read_bytes<uint64_t>(is);
    auto start_time = read_bytes<Tt>(is);
    auto time_step = read_bytes<Tt>(is);
    auto interp = static_cast<InterpolationMode>(read_bytes<uint8_t>(is));

    std::vector<Tt> times(n_times);
    for (auto& t : times)
      t = read_bytes<Tt>(is);

    std::vector<Tv> values(n_times * n_channels);
    for (auto& v : values)
      v = read_bytes<Tv>(is);

    // Strategy is null after load; materialized lazily from the enum on first eval.
    return TsT(std::move(times), std::move(values),
               n_channels, start_time, time_step, interp);
  }
}

#endif // MASSPCF_TIMESERIES_IO_H
