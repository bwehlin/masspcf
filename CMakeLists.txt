cmake_minimum_required(VERSION 3.20...3.26)

if (APPLE)
  set(BUILD_WITH_CUDA OFF)
else()
  set(BUILD_WITH_CUDA ON)
endif()

set(BUILD_TESTER ON)
set(BUILD_BENCHMARKS OFF)

if (SKBUILD)
  set(BUILD_TESTER OFF)
  set(BUILD_BENCHMARKS OFF)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (BUILD_WITH_CUDA)
  SET(LANG_CUDA CUDA)
else()
  SET(LANG_CUDA )
endif()

if (NOT SKBUILD)
  set(SKBUILD_PROJECT_NAME massivepcf)
  set(SKBUILD_PROJECT_VERSION "0.0.1")
endif()

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX ${LANG_CUDA})

if (SKBUILD)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
else()
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Embed)
endif()

include_directories("taskflow")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(pybind11 CONFIG)
if (NOT ${pybind11_FOUND})
  add_subdirectory(pybind11)
endif() 

if (BUILD_WITH_CUDA)
  SET(MPCF_LIB_SOURCES_CUDA
    include/mpcf/cuda/cuda_matrix_integrate.cuh
    include/mpcf/cuda/cuda_util.cuh
    include/mpcf/cuda/cuda_device_array.cuh
    include/mpcf/cuda/cuda_functional_support.cuh
    include/mpcf/cuda/cuda_matrix_integrate_structs.cuh
  )
else()
  SET(MPCF_LIB_SOURCES_CUDA "")
endif()

SET(MPCF_LIB_SOURCES
  include/mpcf/pcf.h
  include/mpcf/point.h
  include/mpcf/rectangle.h
  include/mpcf/task.h
  include/mpcf/executor.h src/executor.cpp
  include/mpcf/algorithm.h
  include/mpcf/algorithms/iterate_rectangles.h
  include/mpcf/algorithms/reduce.h
  include/mpcf/algorithms/matrix_integrate.h
  include/mpcf/algorithms/cuda_matrix_integrate.h
  include/mpcf/algorithms/subdivide.h
  src/block_matrix_support.h
  ${MPCF_LIB_SOURCES_CUDA}
)

SET(MPCF_PY_SOURCES 
  src/python/pymodule.cpp
  src/python/pypcf_support.h
)

if (BUILD_WITH_CUDA)
set_source_files_properties(src/python/pymodule.cpp PROPERTIES LANGUAGE CUDA)
endif()

if (SKBUILD)
  python_add_library(mpcf_cpp MODULE ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES} WITH_SOABI)
else()
  python_add_library(mpcf_cpp SHARED ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES})
endif()

if (BUILD_WITH_CUDA)
  SET(CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(mpcf_cpp PRIVATE -D BUILD_WITH_CUDA=1)
endif()

target_include_directories(mpcf_cpp PRIVATE "include/mpcf")

target_include_directories(mpcf_cpp PUBLIC "include/")
target_link_libraries(mpcf_cpp PRIVATE pybind11::headers)
target_compile_definitions(mpcf_cpp PRIVATE VERSION_INFO=${PROJECT_VERSION})

if (NOT SKBUILD AND BUILD_TESTER)
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

if (BUILD_BENCHMARKS)
  set(Boost_USE_STATIC_LIBS   ON)
  set(Boost_USE_MULTITHREADED ON)
  
  find_package(Boost COMPONENTS program_options system REQUIRED)
  find_package(Threads REQUIRED)
  
  add_executable(mpcf_bench
    benchmarking/bench.cpp
    benchmarking/benchmark.h
    benchmarking/bench_reduction.h benchmarking/bench_reduction.cpp
    )
  
  target_link_libraries(mpcf_bench PUBLIC mpcf_cpp ${CUDART_LIBRARY} ${Boost_LIBRARIES} Threads::Threads)
endif()

install(TARGETS mpcf_cpp DESTINATION mpcf)
