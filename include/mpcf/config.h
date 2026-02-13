/*
* Copyright 2024-2026 Bjorn Wehlin
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

#ifndef MASSPCF_CONFIG_H
#define MASSPCF_CONFIG_H

// ptrdiff_t/size_t user-defined literals. This will come in C++23 (as, e.g., uz instead of _uz)
[[nodiscard]] inline size_t operator""_uz(size_t v) noexcept
{
  return v;
}

[[nodiscard]] inline ptrdiff_t operator""_z(unsigned long long v) noexcept
{
  return static_cast<ptrdiff_t>(v);
}

#include <concepts>
#include <type_traits>

namespace mpcf
{
  template<typename T>
  concept Arithmetic = std::is_arithmetic_v<T>;

  template <typename T>
  concept Iterable = requires(T t) {
    std::begin(t);
    std::end(t);
  };
}

#endif //MASSPCF_CONFIG_H