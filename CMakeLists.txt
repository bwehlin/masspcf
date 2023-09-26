cmake_minimum_required(VERSION 3.20...3.26)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(masspcf LANGUAGES CXX CUDA)

SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA 12.0 REQUIRED)

include_directories("taskflow")

add_library(mpcf SHARED 
  include/mpcf/pcf.h
  include/mpcf/point.h
  include/mpcf/rectangle.h
  include/mpcf/algorithm.h
  include/mpcf/algorithms/iterate_rectangles.h
  include/mpcf/algorithms/reduce.h
  include/mpcf/algorithms/cuda_matrix_integrate.h
  src/cuda_matrix_integrate.cu
  src/cuda_util.h
  src/cuda_device_array.h
  src/block_matrix_support.h)

target_include_directories(mpcf PRIVATE "include/mpcf")

add_subdirectory(pybind11)
pybind11_add_module(mpcf_py
  src/python/pymodule.cpp
  src/python/pypcf_support.h
  )

target_include_directories(mpcf_py PRIVATE "pybind11/include" "include/")
target_link_libraries(mpcf_py PRIVATE mpcf)

######################################################################
# From: https://google.github.io/googletest/quickstart-cmake.html
######################################################################
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
######################################################################

add_executable(mpcf_test
  test/block_matrix_support.cpp)
target_include_directories(mpcf_test PRIVATE "include/")

target_link_libraries(mpcf_test GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(mpcf_test)




# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(mpcf_py
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
