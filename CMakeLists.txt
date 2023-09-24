cmake_minimum_required(VERSION 3.4...3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(mpcf_cpp)

include_directories("pybind11/include")

add_subdirectory(pybind11)
pybind11_add_module(mpcf_cpp src/python/pymodule.cpp)




# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(mpcf_cpp
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
