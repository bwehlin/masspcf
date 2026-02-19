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


#ifndef MASSPCF_PY_IO_H
#define MASSPCF_PY_IO_H

#include <pybind11/pybind11.h>
#include <iostream>
#include <streambuf>
#include <string>

namespace py = pybind11;

namespace mpcf_py
{
  class PythonBuf : public std::streambuf {
  public:
    explicit PythonBuf(py::object py_file, size_t buffer_size = 1024)
        : py_file(std::move(py_file)), buffer(buffer_size) {
      // Set up the internal buffer pointers: [begin, current, end]
      setp(buffer.data(), buffer.data() + buffer.size());
    }

  protected:
    // Called when the buffer is full or flushed
    int sync() override {
      if (pbase() != pptr()) {
        py::gil_scoped_acquire acquire; // Ensure we have the GIL
        std::string str(pbase(), pptr());
        py_file.attr("write")(str);
        setp(buffer.data(), buffer.data() + buffer.size());
      }
      return 0;
    }

    // Called when a character is written to a full buffer
    int_type overflow(int_type ch) override {
      if (sync() == -1) return traits_type::eof();
      if (ch != traits_type::eof()) {
        *pptr() = traits_type::to_char_type(ch);
        pbump(1);
      }
      return ch;
    }

  private:
    py::object py_file;
    std::vector<char> buffer;
  };

  void register_io(pybind11::module_& m);
}

#endif //MASSPCF_PY_IO_H
