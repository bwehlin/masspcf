// Header-only replacement for the original Ripser++ phmap_interface shim.
// The upstream version is a .cpp that declares a single file-scope
// parallel_flat_hash_map<int64_t, int64_t> and three free functions.
// Making it header-only (inline variable) lets C++ consumers use only the
// mpcf include directory without a separate compile step.
//
// NOTE: the state here is global and therefore not thread-safe across
// concurrent Ripser++ invocations. Removing this global is part of the
// refactor needed before multiple GPU jobs can run simultaneously.

#ifndef MASSPCF_RIPSERPP_PHMAP_INTERFACE_HPP
#define MASSPCF_RIPSERPP_PHMAP_INTERFACE_HPP

#include <cstdint>

#include "../../internal/parallel_hashmap/phmap.h"

namespace mpcf::ripserpp::detail
{
  inline phmap::parallel_flat_hash_map<int64_t, int64_t>& phmap_interface_state()
  {
    static phmap::parallel_flat_hash_map<int64_t, int64_t> state;
    return state;
  }
}

inline void phmap_put(int64_t key, int64_t value)
{
  mpcf::ripserpp::detail::phmap_interface_state()[key] = value;
}

inline int64_t phmap_get_value(int64_t key)
{
  auto& s = mpcf::ripserpp::detail::phmap_interface_state();
  auto it = s.find(key);
  if (it != s.end())
  {
    return it->second;
  }
  return -1;
}

inline void phmap_clear()
{
  mpcf::ripserpp::detail::phmap_interface_state().clear();
}

#endif // MASSPCF_RIPSERPP_PHMAP_INTERFACE_HPP
