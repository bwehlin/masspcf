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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <type_traits>
#include <typeinfo>
#include <memory>

#if defined(__clang__) || defined(__GNUC__)
#include <cxxabi.h> // For name demangling
#endif

namespace mpcf
{
  using float32_t = _Float32;
  using float64_t = _Float64;

  using int32_t = std::int32_t;
  using int64_t = std::int64_t;
  using uint32_t = std::uint32_t;
  using uint64_t = std::uint64_t;

  template <typename T>
  concept FloatType = std::is_floating_point_v<T>
    || std::is_same_v<T, _Float32>
    || std::is_same_v<T, _Float64>;

  template <typename T>
  concept UnsignedIntType =
    (std::is_integral_v<T> && std::is_unsigned_v<T>)
    || std::is_same_v<T, uint32_t>
    || std::is_same_v<T, uint64_t>;

  template <typename T>
  concept SignedIntType =
    (std::is_integral_v<T> && std::is_signed_v<T>)
    || std::is_same_v<T, int32_t>
    || std::is_same_v<T, int64_t>;

  template <typename T>
  concept IntType = UnsignedIntType<T> || SignedIntType<T>;

  template <typename T>
  concept ArithmeticType = FloatType<T> || IntType<T>;

}

// Note: we do this for now. These should be true on basically all platforms that we care about. We may have to modify
// this assumption in the future.
static_assert(sizeof(double) == sizeof(_Float64), "double and _Float64 must be the same size");
static_assert(sizeof(float)  == sizeof(_Float32), "float and _Float32 must be the same size");

// This is *technically* UB, but we'll live with this to make our lives a bit easier rather than having to cast what's
// a 'float' in practice to float, etc.
template<> struct std::is_floating_point<mpcf::float32_t> : std::true_type {};
template<> struct std::is_floating_point<mpcf::float64_t> : std::true_type {};
template<> struct std::numeric_limits<mpcf::float32_t> : std::numeric_limits<float> { };
//template<> struct std::numeric_limits<mpcf::float64_t> : std::numeric_limits<double> { };

template<>
struct std::numeric_limits<_Float64> : std::numeric_limits<double> {};

// ptrdiff_t/size_t user-defined literals. This will come in C++23 (as, e.g., uz instead of _uz)
[[nodiscard]] inline size_t operator""_uz(unsigned long long v) noexcept
{
  return v;
}

[[nodiscard]] inline ptrdiff_t operator""_z(unsigned long long v) noexcept
{
  return static_cast<ptrdiff_t>(v);
}

namespace mpcf
{
  template<typename T>
  concept Arithmetic = std::is_arithmetic_v<T>;

  template <typename T>
  concept Iterable = requires(T t) {
    std::begin(t);
    std::end(t);
  };

  namespace detail
  {
    template <typename T>
    std::string unmangled_typename()
    {
#if defined(__clang__) || defined(__GNUC__)
      int status;
      std::unique_ptr<char, void(*)(void*)> name(
        abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status),
        std::free
      );
      return (status == 0) ? name.get() : typeid(T).name();
#else
      return typeid(T).name(); // Should be demangled at least on MSVC
#endif
    }
  }
}

#endif //MASSPCF_CONFIG_H