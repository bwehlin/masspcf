cmake_minimum_required(VERSION 3.20...3.26)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX CUDA)

if (SKBUILD)  
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
else()
  set(SKBUILD_PROJECT_NAME massivepcf)
  set(SKBUILD_PROJECT_VERSION "0.0.1")
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Embed)
endif()

SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA 12.0 REQUIRED)

include_directories("taskflow")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(pybind11 CONFIG)
if (NOT ${pybind11_FOUND})
  add_subdirectory(pybind11)
endif() 

SET(MPCF_LIB_SOURCES
  include/mpcf/pcf.h
  include/mpcf/point.h
  include/mpcf/rectangle.h
  include/mpcf/algorithm.h
  include/mpcf/algorithms/iterate_rectangles.h
  include/mpcf/algorithms/reduce.h
  include/mpcf/algorithms/matrix_integrate.h
  include/mpcf/algorithms/cuda_matrix_integrate.h
  src/cuda_matrix_integrate.cu
  src/cuda_util.h
  src/cuda_device_array.h
  src/cuda_functional_support.h
  src/cuda_matrix_integrate_structs.h
  src/block_matrix_support.h
)

SET(MPCF_PY_SOURCES 
  src/python/pymodule.cpp
  src/python/pypcf_support.h
)

if (SKBUILD)
  python_add_library(mpcf_cpp MODULE ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES} WITH_SOABI)
else()
  python_add_library(mpcf_cpp SHARED ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES})
endif()

#add_library(mpcf SHARED ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES})

target_include_directories(mpcf_cpp PRIVATE "include/mpcf")
 
#add_subdirectory(pybind11)



#pybind11_add_module(mpcf_py
#  src/python/pymodule.cpp
#  src/python/pypcf_support.h
#  )

target_include_directories(mpcf_cpp PUBLIC "include/")
target_link_libraries(mpcf_cpp PRIVATE pybind11::headers)
target_compile_definitions(mpcf_cpp PRIVATE VERSION_INFO=${PROJECT_VERSION})

if (NOT SKBUILD)
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
    test/block_matrix_support.cpp
    test/basic_l1_dist.cpp)
  target_include_directories(mpcf_test PRIVATE "include/")

  target_link_libraries(mpcf_test PRIVATE GTest::gtest_main mpcf_cpp ${CUDART_LIBRARY})
  include(GoogleTest)
  gtest_discover_tests(mpcf_test)
endif()

#set_target_properties(mpcf_cpp PROPERTIES SUFFIX "${PYTHON_EXTENSION_MODULE_SUFFIX}")
install(TARGETS mpcf_cpp DESTINATION mpcf)
