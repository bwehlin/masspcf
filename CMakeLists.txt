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
  src/pcf.h
  src/point.h
  src/pypcf_support.h
  src/rectangle.h
  src/algorithm.h
  src/algorithms/iterate_rectangles.h
  src/algorithms/reduce.h
  src/algorithms/cuda_matrix_integrate.h
  src/algorithms/cuda_matrix_integrate.cu)




# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(mpcf_cpp
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
