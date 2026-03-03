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

#ifndef MASSPCF_PYBIND_H
#define MASSPCF_PYBIND_H

#include <mpcf/config.h>

#include <pybind11/pybind11.h>

namespace pybind11::detail
{
  // Probably not needed any more
#if 0
  template <> class type_caster<mpcf::float32_t>
  {
  public:
    PYBIND11_TYPE_CASTER(mpcf::float32_t, const_name("float"));

    bool load(handle src, bool convert)
    {
      if (!PyFloat_Check(src.ptr()))
      {
        if (!convert)
          return false;
        PyObject *tmp = PyNumber_Float(src.ptr());
        if (!tmp)
        {
          PyErr_Clear();
          return false;
        }
        value = static_cast<mpcf::float32_t>(PyFloat_AsDouble(tmp));
        Py_DECREF(tmp);
      }
      else
      {
        value = static_cast<mpcf::float32_t>(PyFloat_AsDouble(src.ptr()));
      }
      return !PyErr_Occurred();
    }

    static handle cast(mpcf::float32_t src, return_value_policy, handle)
    {
      return PyFloat_FromDouble(static_cast<double>(src));
    }
  };

  template <> class type_caster<mpcf::float64_t>
  {
  public:
    PYBIND11_TYPE_CASTER(mpcf::float64_t, const_name("float"));

    bool load(handle src, bool convert)
    {
      if (!PyFloat_Check(src.ptr()))
      {
        if (!convert)
          return false;
        PyObject *tmp = PyNumber_Float(src.ptr());
        if (!tmp)
        {
          PyErr_Clear();
          return false;
        }
        value = static_cast<mpcf::float64_t>(PyFloat_AsDouble(tmp));
        Py_DECREF(tmp);
      }
      else
      {
        value = static_cast<mpcf::float64_t>(PyFloat_AsDouble(src.ptr()));
      }
      return !PyErr_Occurred();
    }

    static handle cast(mpcf::float64_t src, return_value_policy, handle)
    {
      return PyFloat_FromDouble(static_cast<double>(src));
    }
  };
#endif
} // namespace pybind11::detail

#endif // MASSPCF_PYBIND_H
