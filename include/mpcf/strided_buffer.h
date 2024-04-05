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

#ifndef MPCF_STRIDED_BUFFER_H
#define MPCF_STRIDED_BUFFER_H

#include <vector>
#include <iterator>

namespace mpcf
{
  template <typename T>
  struct StridedBuffer
  {
    T* buffer;
    std::vector<size_t> strides;
    std::vector<size_t> shape;

    class Iterator
    {
    public:
      using difference_type = std::ptrdiff_t;
      using reference = T&;
      using pointer = T*;
      using value_type = T;
      using iterator_category = std::random_access_iterator_tag;

      Iterator& operator++()
      {
        ++m_n;
        return *this;
      }

      Iterator& operator++(int)
      {
        Iterator ret = *this;
        ++(*this);
        return ret;
      }

      Iterator& operator--()
      {
        --m_n;
        return *this;
      }

      Iterator& operator--(int)
      {
        Iterator ret = *this;
        --(*this);
        return ret;
      }

      reference operator*() const
      {
        return *(m_buffer + m_n * m_stride);
      }

      pointer operator->() const
      {
        return m_buffer + m_n * m_stride;
      }

      reference operator[](size_t n) const
      {
        return *(*this + n);
      }

      bool operator==(const Iterator& rhs) const
      {
        return m_n == rhs.m_n;
      }

      bool operator!=(const Iterator& rhs) const
      {
        return m_n != rhs.m_n;
      }

      bool operator<(const Iterator& rhs) const
      {
        return m_n < rhs.m_n;
      }

      bool operator<=(const Iterator& rhs) const
      {
        return m_n <= rhs.m_n;
      }

      bool operator>(const Iterator& rhs) const
      {
        return m_n > rhs.m_n;
      }

      bool operator>=(const Iterator& rhs) const
      {
        return m_n >= rhs.m_n;
      }

      Iterator& operator+=(difference_type n)
      {
        m_n += n;
        return *this;
      }

      Iterator& operator-=(difference_type n)
      {
        m_n -= n;
        return *this;
      }

    private:
      friend StridedBuffer;

      T* m_buffer;
      size_t m_stride;
      size_t m_n;
    };

    Iterator begin(size_t axis)
    {
      Iterator it;
      it.m_buffer = buffer;
      it.m_stride = strides[axis];
      it.m_n = 0;
    }

    Iterator end(size_t axis)
    {
      Iterator it;
      it.m_buffer = buffer;
      it.m_stride = strides[axis];
      it.m_n = shape[axis] + 1;
    }

  };

}

#endif
