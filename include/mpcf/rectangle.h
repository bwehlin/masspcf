/*
* Copyright 2024 Bjorn Wehlin
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

#ifndef MPCF_RECTANGLE_H
#define MPCF_RECTANGLE_H

#include <ostream>

namespace mpcf
{
  template <typename Tt, typename Tv>
  struct Rectangle
  {
    Tt left = 0;
    Tt right = 0;
    Tv top = 0;
    Tv bottom = 0;

    Rectangle() = default;
    Rectangle(Tt l, Tt r, Tv t, Tv b)
      : left(l), right(r), top(t), bottom(b)
    { }
    
    Rectangle& l(Tt t)
    {
      left = t;
      return *this;
    }
    
    Rectangle& r(Tt t)
    {
      right = t;
      return *this;
    }
    
    Rectangle& fv(Tv v)
    {
      top = v;
      return *this;
    }
    
    Rectangle& gv(Tv v)
    {
      bottom = v;
      return *this;
    }
    
    bool operator==(const Rectangle& rhs) const
    {
      return left == rhs.left && right == rhs.right && top == rhs.top && bottom == rhs.bottom;
    }
    
    bool operator!=(const Rectangle& rhs) const
    {
      return left != rhs.left || right != rhs.right || top != rhs.top || bottom != rhs.bottom;
    }
    
    template <typename Ut, typename Uv>
    friend std::ostream& operator<<(std::ostream&, const Rectangle<Ut, Uv>&);
  };

  template <typename Tt, typename Tv>
  struct Segment
  {
    Tt left = 0;
    Tt right = 0;
    Tv value = 0;

    Segment() = default;
    Segment(Tt l, Tt r, Tv v)
      : left(l), right(r), value(v)
    { }
  };
  
  template <typename Tt, typename Tv>
  std::ostream& operator<<(std::ostream& os, const mpcf::Rectangle<Tt, Tv>& rect)
  {
    os << "Rectangle(.l = " << rect.left << ", .r = " << rect.right << ", .fv = " << rect.top << ", .gv = " << rect.bottom << ")";
    return os;
  }
}

#endif
