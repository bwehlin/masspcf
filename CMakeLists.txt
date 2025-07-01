# Copyright 2024-2025 Bjorn Wehlin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.16)

include(FetchContent)

if((DEFINED ENV{MINIMAL_MODULE_BUILD}) AND (NOT $ENV{MINIMAL_MODULE_BUILD} MATCHES "0"))
  set(MINIMAL_MODULE_BUILD ON)
  message(STATUS "!!! Minimal module build !!!")
else()
  message(STATUS "!!! Not Minimal module build !!!")
  set(MINIMAL_MODULE_BUILD OFF)
endif()

if((DEFINED ENV{BUILD_WITH_CUDA}) AND (NOT $ENV{BUILD_WITH_CUDA} MATCHES "0"))
  if (APPLE)
    set(BUILD_WITH_CUDA OFF)
  else()
    set(BUILD_WITH_CUDA ON)
  endif()
else()
  set(BUILD_WITH_CUDA OFF)
endif()

set(BUILD_TESTER ON)
set(BUILD_BENCHMARKS OFF)

if (SKBUILD OR MINIMAL_MODULE_BUILD)
  set(BUILD_TESTER OFF)
  set(BUILD_BENCHMARKS OFF)
endif()

if (BUILD_WITH_CUDA)
  message(STATUS "!!! Building with CUDA !!!")

  SET(LANG_CUDA CUDA)
  add_compile_definitions(BUILD_WITH_CUDA=1)
else()
  message(STATUS "!!! Building without CUDA !!!")

  SET(LANG_CUDA )
endif()

if (BUILD_WITH_CUDA)
  # It seemed easier to use .cpp and compile as CUDA than using .cu and compiling as CXX (when CUDA support is disabled)
  set_source_files_properties(src/python/pymodule.cpp PROPERTIES LANGUAGE CUDA)
endif()

if (NOT SKBUILD AND NOT MINIMAL_MODULE_BUILD)
  set(SKBUILD_PROJECT_NAME massivepcf)
  set(SKBUILD_PROJECT_VERSION "0.3.2")
endif()

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX ${LANG_CUDA})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CXX)

if (WIN32)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Development.Embed Development.Module)

include_directories("3rd/taskflow")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


add_subdirectory(3rd/pybind11)

set(xtl_DIR "3rd/xtl")
set(xtensor_DIR "3rd/xtensor")

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
  include/mpcf/strided_buffer.h
  include/mpcf/operations.cuh
  include/mpcf/platform.h
  include/mpcf/random.h
  include/mpcf/task.h
  include/mpcf/executor.h src/executor.cpp
  include/mpcf/algorithm.h
  include/mpcf/algorithms/iterate_rectangles.h
  include/mpcf/algorithms/reduce.h
  include/mpcf/algorithms/matrix_integrate.h
  include/mpcf/algorithms/matrix_reduce.h
  include/mpcf/algorithms/subdivide.h
  include/mpcf/algorithms/apply_functional.h
  include/mpcf/block_matrix_support.cuh
  ${MPCF_LIB_SOURCES_CUDA}
)

SET(MPCF_PY_SOURCES 
  src/python/pyarray.h src/python/pyarray.cpp
  src/python/pymodule.cpp
  src/python/pyrandom.cpp
  src/python/pypcf_support.h
  src/python/py_make_from_serial_content.h src/python/py_make_from_serial_content.cpp
)



if (SKBUILD OR MINIMAL_MODULE_BUILD)
  python_add_library(mpcf_cpp MODULE ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES} WITH_SOABI)
else()
  python_add_library(mpcf_cpp SHARED ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES})
endif()

if (BUILD_WITH_CUDA)
  SET(CUDA_SEPARABLE_COMPILATION ON)
endif()

target_include_directories(mpcf_cpp PRIVATE "include/mpcf")
target_include_directories(mpcf_cpp PUBLIC "include/")

include_directories("3rd/xtensor/include" "3rd/xtl/include")
target_link_libraries(mpcf_cpp PRIVATE pybind11::headers)
target_compile_definitions(mpcf_cpp PRIVATE VERSION_INFO=${PROJECT_VERSION}) 

if (MSVC)
  # MSVC CUDA has some problems with picking up include directories in IntelliSense. We can solve this
  # by using VC++ Include Directories instead of on the compiler-level
  get_filename_component(taskflow_DIR "3rd/taskflow" REALPATH)
  set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$(VC_IncludePath);$(WindowsSDK_IncludePath);${taskflow_DIR};${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

if (NOT (SKBUILD OR MINIMAL_MODULE_BUILD) AND BUILD_TESTER)
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
    test/test_iterate_rectangles.cpp
    test/test_l1_dist.cu
    test/test_norms.cu
    test/test_parallel_reduce.cpp)
  target_include_directories(mpcf_test PRIVATE "include/")
  target_include_directories(mpcf_test SYSTEM PRIVATE "3rd/xtensor/include" "3rd/xtl/include")

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

set_target_properties(mpcf_cpp PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

if(MSVC)
  target_compile_options(mpcf_cpp PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>")
else()
  #target_compile_options(mpcf_cpp PRIVATE -Wall -Wextra)
  set( CMAKE_CXX_FLAGS "-Wall -Wextra" )
endif()




if (BUILD_WITH_CUDA)

  find_package(CUDAToolkit REQUIRED)
  message("CUDART: ${CUDAToolkit_LIBRARY_DIR}")
  set(CUDART_LIB ${CUDAToolkit_LIBRARY_DIR}/libcudart.so)
  file(GLOB cudart_libs ${CUDAToolkit_LIBRARY_DIR}/libcudart.so*)
  message("cudart_libs: ${cudart_libs}")
  
  install(FILES ${cudart_libs} DESTINATION masspcf)

endif()

if (MINIMAL_MODULE_BUILD)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
      import site
      print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  SET(SITE_PKG_MASSPCF ${PYTHON_SITE}/masspcf)

  message(STATUS "Minimal module will be installed into ${SITE_PKG_MASSPCF}")

  file(GLOB PY_PKG_FILES "masspcf/*.py")
  #message(STATUS ${PY_PKG_FILES})
  install(
    DIRECTORY masspcf
    DESTINATION ${PYTHON_SITE}
    PATTERN "__pycache__" EXCLUDE
    )
  
  install(
    TARGETS mpcf_cpp
    DESTINATION ${SITE_PKG_MASSPCF}
  )
else()
  install(TARGETS mpcf_cpp DESTINATION masspcf)
endif()
