#ifndef MPCF_REDUCE_OPS_H
#define MPCF_REDUCE_OPS_H

#include "rectangle.h"

namespace mpcf
{
  template <typename PcfT>
  struct add
  {
    using time_type = typename PcfT::time_type;
    using value_type = typename PcfT::value_type;
    using rectangle_type = typename PcfT::rectangle_type;
    
    value_type operator()(const rectangle_type& rect) const noexcept
    {
      return rect.top + rect.bottom;
    }
  };
}

#endif
