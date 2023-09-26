cmake_minimum_required(VERSION 3.4...3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(mpcf_cpp LANGUAGES CXX CUDA)

SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA 12.0 REQUIRED)

include_directories("pybind11/include")
include_directories("taskflow")

add_subdirectory(pybind11)
pybind11_add_module(mpcf_cpp 
  src/python/pymodule.cpp 
  include/pcf.h
  include/point.h
  include/pypcf_support.h
  include/rectangle.h
  include/algorithm.h
  include/algorithms/iterate_rectangles.h
  include/algorithms/reduce.h
  include/algorithms/cuda_matrix_integrate.h
  src/cuda_matrix_integrate.cu
  src/cuda_util.h
  src/cuda_device_array.h
  src/block_matrix_support.h)

include_directories("include")



# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(mpcf_cpp
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
