#ifndef MPCF_ARRAY_H
#define MPCF_ARRAY_H

#include <memory>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "pcf.h"

namespace mpcf
{
  template <typename Tt, typename Tv>
  class Array
  {
  public:
    using value_type = mpcf::Pcf<Tt, Tv>;
    using reference = mpcf::Pcf<Tt, Tv>&;
    using const_reference = const mpcf::Pcf<Tt, Tv>&;
    
    Array()
    {
      
    }
    
    Array(const std::vector<size_t> dims)
      : m_shape(dims)
    {
      init_strides(dims);
      auto sz = get_linear_size();
      m_data = std::make_unique<value_type[]>(sz);
    }
    
    Array(const Array& other)
      : m_shape(other.m_shape)
      , m_strides(other.m_strides)
    {
      auto sz = get_linear_size();
      m_data = std::make_unique<value_type[]>(sz);
      for (size_t i = 0; i < sz; ++i)
      {
        m_data[i] = other.m_data[i];
      }
    }
    
    Array(Array&& other) noexcept
      : m_shape(std::move(other.m_shape))
      , m_strides(std::move(other.m_strides))
      , m_data(std::move(other.m_data))
    { }
    
    Array& operator=(const Array& rhs)
    {
      if (this == &rhs)
      {
        return *this;
      }
      
      m_shape = rhs.m_shape;
      m_strides = rhs.m_strides;
      auto sz = get_linear_size();
      for (size_t i = 0; i < sz; ++i)
      {
        m_data[i] = rhs.m_data[i];
      }
      
      return *this;
    }
    
    Array& operator=(Array&& rhs) noexcept
    {
      m_shape = std::move(rhs.m_shape);
      m_strides = std::move(rhs.m_strides);
      m_data = std::move(rhs.m_data);
      
      return *this;
    }
    
    [[nodiscard]] const std::vector<size_t>& strides() const noexcept 
    {
      return m_strides;
    }
    
    [[nodiscard]] const std::vector<size_t>& shape() const noexcept 
    {
      return m_shape;
    }
    
#if 0
    [[nodiscard]] const_reference at(const std::vector<size_t>& pos) const
    {
      if (pos.size() != m_shape.size())
      {
        throw std::runtime_error("'pos' must have the same number of elements as 'shape'.");
      }
      
      auto index = std::inner_product(pos.begin(), pos.end(), m_strides.begin(), 0, std::plus<>(), std::multiplies<>());
    }
#endif
    
    [[nodiscard]] size_t get_linear_index(const std::vector<size_t>& pos) const noexcept
    {
      return std::inner_product(pos.begin(), pos.end(), m_strides.begin(), 0, std::plus<>(), std::multiplies<>());
    }
    
    [[nodiscard]] size_t get_linear_size() const noexcept
    {
      return get_linear_size(m_shape);
    }
    
  private:
    
    [[nodiscard]] size_t get_linear_size(const std::vector<size_t>& dims) const noexcept
    {
      return std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
    }
    
    void init_strides(const std::vector<size_t> dims)
    {
      if (dims.empty())
      {
        return;
      }
      
      m_strides.resize(dims.size());
      std::exclusive_scan(dims.rbegin(), dims.rend(), m_strides.rbegin(), 1, std::multiplies<>{});
    }
    
    std::unique_ptr<value_type[]> m_data;
    std::vector<size_t> m_shape;
    std::vector<size_t> m_strides;
    
  };
  
  using Array_f32 = Array<float, float>;
  using Array_f64 = Array<double, double>;
}

#endif
