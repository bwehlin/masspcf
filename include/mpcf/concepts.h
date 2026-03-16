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

#ifndef MASSPCF_CONCEPTS_H
#define MASSPCF_CONCEPTS_H

#include <concepts>

namespace mpcf
{
  template <typename T, typename U>
  concept CanDivide = requires(T t, U u)
  {
    t / u;
  };

  template <typename T, typename U>
  concept CanMultiply = requires(T t, U u)
  {
    t * u;
  };

  template <typename T, typename U>
  concept CanAdd = requires(T t, U u)
  {
    t + u;
  };

  template <typename T, typename U>
  concept CanSubtract = requires(T t, U u)
  {
    t - u;
  };

  // "Into" variants: check that the result of A op B is convertible to R.
  // Use these for operators whose body stores the result back into an element
  // of type R (e.g. compound assignment, elementwise free operators).

  template <typename R, typename A, typename B>
  concept CanAddTo = requires(A a, B b)
  {
    { a + b } -> std::convertible_to<R>;
  };

  template <typename R, typename A, typename B>
  concept CanSubtractTo = requires(A a, B b)
  {
    { a - b } -> std::convertible_to<R>;
  };

  template <typename R, typename A, typename B>
  concept CanMultiplyTo = requires(A a, B b)
  {
    { a * b } -> std::convertible_to<R>;
  };

  template <typename R, typename A, typename B>
  concept CanDivideTo = requires(A a, B b)
  {
    { a / b } -> std::convertible_to<R>;
  };
}

#endif //MASSPCF_CONCEPTS_H
