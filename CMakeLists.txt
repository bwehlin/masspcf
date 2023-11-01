cmake_minimum_required(VERSION 3.20...3.27)

include(FetchContent)

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

if (WIN32)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif() 

if (BUILD_WITH_CUDA)
  SET(LANG_CUDA CUDA)
  add_compile_definitions(BUILD_WITH_CUDA=1)
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

message(STATUS "Python executable: ${Python_EXECUTABLE} ")


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
  include/mpcf/operations.cuh
  include/mpcf/platform.h
  include/mpcf/task.h
  include/mpcf/executor.h src/executor.cu
  include/mpcf/algorithm.h
  include/mpcf/algorithms/iterate_rectangles.h
  include/mpcf/algorithms/reduce.h
  include/mpcf/algorithms/matrix_integrate.h
  include/mpcf/algorithms/subdivide.h
  include/mpcf/algorithms/permute_in_place.cuh
  include/mpcf/block_matrix_support.cuh
  ${MPCF_LIB_SOURCES_CUDA}
)

SET(MPCF_PY_SOURCES 
  src/python/pymodule.cu
  src/python/pypcf_support.h
)

if (NOT BUILD_WITH_CUDA)
	set_source_files_properties(src/python/pymodule.cu PROPERTIES LANGUAGE CXX)
  set_source_files_properties(src/executor.cu PROPERTIES LANGUAGE CXX)
endif()

if (SKBUILD)
  python_add_library(mpcf_cpp MODULE ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES} WITH_SOABI)
else()
  python_add_library(mpcf_cpp SHARED ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES})
endif()

if (BUILD_WITH_CUDA)
  SET(CUDA_SEPARABLE_COMPILATION ON)
endif()

target_include_directories(mpcf_cpp PRIVATE "include/mpcf")

target_include_directories(mpcf_cpp PUBLIC "include/")
target_link_libraries(mpcf_cpp PRIVATE pybind11::headers)
target_compile_definitions(mpcf_cpp PRIVATE VERSION_INFO=${PROJECT_VERSION}) 

if (NOT SKBUILD AND BUILD_TESTER)
  set(BUILD_GMOCK OFF)
  set(INSTALL_GTEST OFF)
  set(gtest_force_shared_crt ON)

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
  )
  
  FetchContent_GetProperties(googletest)
  if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
  endif()

  add_executable(mpcf_test
    test/test_block_matrix_support.cu
    test/test_l1_dist.cu
    test/test_permutations.cu
    test/test_permute_in_place.cu)
  target_include_directories(mpcf_test PRIVATE "include/")

  target_link_libraries(mpcf_test PUBLIC gtest_main mpcf_cpp ${CUDART_LIBRARY})
  
  #include(GoogleTest)
  #gtest_discover_tests(mpcf_test)

  if (WIN32)
    get_filename_component(python_dir "${Python_EXECUTABLE}" DIRECTORY)
    set_target_properties(mpcf_test PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=${python_dir}")
  endif()
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

#set_target_properties(mpcf_cpp PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

if (WIN32) 
  #set_property(TARGET mpcf_cpp PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  #set_property(TARGET mpcf_test PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  #set_property(TARGET gtest_main PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  #set_property(TARGET gtest PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  #set_property(TARGET mpcf_bench PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif() 



install(TARGETS mpcf_cpp DESTINATION mpcf)
