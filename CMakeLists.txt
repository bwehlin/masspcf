# Copyright 2024-2026 Bjorn Wehlin
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

cmake_minimum_required(VERSION 3.19)

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

if (SKBUILD)
  set(BUILD_TESTER OFF)
else()
  set(BUILD_TESTER ON)
endif()

set(BUILD_BENCHMARKS OFF)

if (SKBUILD OR MINIMAL_MODULE_BUILD)
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

if (SKBUILD OR MINIMAL_MODULE_BUILD)
  set(PROJ_NAME masspcf)
else()
  set(PROJ_NAME massivepcf)
endif()

# Read version info out of pyproject.toml
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" TOML_CONTENT)

# 1. Find the [project] section
string(FIND "${TOML_CONTENT}" "[project]" PROJECT_SECTION_START)

# 2. Slice the string from [project] onwards to avoid [build-system] versions
string(SUBSTRING "${TOML_CONTENT}" ${PROJECT_SECTION_START} -1 PROJECT_BODY)

# 3. Extract the version string
if(PROJECT_BODY MATCHES [[version[ \t]*=[ \t]*"([^"]+)"]])
  set(PROJ_VERSION "${CMAKE_MATCH_1}")
endif()

project(${PROJ_NAME} VERSION ${PROJ_VERSION} LANGUAGES CXX ${LANG_CUDA})

if((DEFINED ENV{ENABLE_COVERAGE}) AND (NOT $ENV{ENABLE_COVERAGE} MATCHES "0"))
  set(ENABLE_COVERAGE ON)
else()
  set(ENABLE_COVERAGE OFF)
endif()

message(STATUS "!!! PROJECT_NAME: ${PROJECT_NAME}, PROJECT_VERSION: ${PROJECT_VERSION} !!!")

set(CPP_MODULE_NAME "_mpcf_cpp")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CXX)

if (WIN32)
  if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
  else()
    add_compile_options(-Wa,-mbig-obj)
  endif()
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Development.Module Interpreter)

include_directories("3rd/taskflow")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(3rd/pybind11)

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

SET(MPCF_HEADER_FILES
  config.h
  pcf.h
  point.h
  rectangle.h
  operations.cuh
  platform.h
  random.h
  task.h
  tensor.h tensor.tpp
  tensor_to_string.h
  special_tensors.h
  executor.h
  io_stream.h
  algorithm.h
  algorithms/iterate_rectangles.h
  algorithms/reduce.h
  algorithms/matrix_integrate.h
  algorithms/matrix_reduce.h
  algorithms/subdivide.h
  algorithms/apply_functional.h
  block_matrix_support.cuh)

SET(MPCF_SRC_FILES
  executor.cpp
  io_stream.cpp)

list(TRANSFORM MPCF_HEADER_FILES PREPEND "include/mpcf/")
list(TRANSFORM MPCF_SRC_FILES PREPEND "src/")

SET(MPCF_LIB_SOURCES
  ${MPCF_HEADER_FILES}
  ${MPCF_SRC_FILES}
  ${MPCF_LIB_SOURCES_CUDA}
)

SET(MPCF_PY_SOURCES
  pymodule.cpp
  pyrandom.cpp
  pypcf_support.h
  py_np_support.h
  py_make_from_serial_content.h py_make_from_serial_content.cpp
  py_distance.h py_distance.cpp
  py_norms.h py_norms.cpp
  py_reductions.h py_reductions.cpp
  py_tensor.h py_tensor.cpp
  py_io.h py_io.cpp
)

list(TRANSFORM MPCF_PY_SOURCES PREPEND "src/python/")


#pybind11_add_module(${CPP_MODULE_NAME} ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES})

pybind11_add_module(${CPP_MODULE_NAME} MODULE ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES})


#if (SKBUILD OR MINIMAL_MODULE_BUILD)
#python_add_library(${CPP_MODULE_NAME} MODULE ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES} WITH_SOABI)
  #add_library(${CPP_MODULE_NAME} MODULE ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES})
#  pybind11_add_module(${CPP_MODULE_NAME} ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES})
#else()
  #python_add_library(${CPP_MODULE_NAME} SHARED ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES})
  #add_library(${CPP_MODULE_NAME} SHARED ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES})
#  pybind11_add_module(${CPP_MODULE_NAME} ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES})
#endif()

if (BUILD_WITH_CUDA)
  SET(CUDA_SEPARABLE_COMPILATION ON)
endif()

set(MPCF_PRIVATE_INCLUDES "include/mpcf")
set(MPCF_PUBLIC_INCLUDES "include/")

target_include_directories(${CPP_MODULE_NAME} PRIVATE ${MPCF_PRIVATE_INCLUDES})
target_include_directories(${CPP_MODULE_NAME} PUBLIC ${MPCF_PUBLIC_INCLUDES})

target_link_libraries(${CPP_MODULE_NAME} PRIVATE pybind11::headers)
target_compile_definitions(${CPP_MODULE_NAME} PRIVATE VERSION_INFO=${PROJECT_VERSION}) 

if (MSVC)
  # MSVC CUDA has some problems with picking up include directories in IntelliSense. We can solve this
  # by using VC++ Include Directories instead of on the compiler-level
  get_filename_component(taskflow_DIR "3rd/taskflow" REALPATH)
  set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$(VC_IncludePath);$(WindowsSDK_IncludePath);${taskflow_DIR};${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

# ============================================================
# Coverage configuration (GCC / Clang only)
# ============================================================



if(ENABLE_COVERAGE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "Coverage only supported with GCC or Clang")
  endif()

  message(STATUS "!!! Coverage ENABLED !!!")

  # Disable optimization for accurate coverage
  add_library(coverage_flags INTERFACE)

  target_compile_options(coverage_flags INTERFACE
          -O0
          -g
          -fprofile-arcs
          -ftest-coverage
          )

  target_link_libraries(coverage_flags INTERFACE gcov)

  # Avoid inlining & weird optimizer effects
  target_compile_options(coverage_flags INTERFACE
          -fno-inline
          -fno-inline-small-functions
          -fno-default-inline
          )
else()
  message(STATUS "!!! Coverage DISABLED !!!")
endif()

if (BUILD_TESTER)
  set(BUILD_GMOCK OFF)
  set(INSTALL_GTEST OFF)
  set(gtest_force_shared_crt ON)

  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  add_subdirectory("3rd/googletest")

  enable_testing()
  add_executable(mpcf_test
    test/test_block_matrix_support.cu
    test/test_iterate_rectangles.cpp
    test/test_l1_dist.cu
    test/test_norms.cu
    test/test_parallel_reduce.cpp
    test/test_tensor_iteration.cpp
    test/test_special_tensors.cpp
    test/test_io_stream.cpp)

  target_include_directories(mpcf_test PRIVATE "include/")

  add_library(mpcf_nopy STATIC ${MPCF_LIB_SOURCES})
  target_include_directories(mpcf_nopy PRIVATE ${MPCF_PRIVATE_INCLUDES})
  target_include_directories(mpcf_nopy PUBLIC ${MPCF_PUBLIC_INCLUDES})

  target_link_libraries(mpcf_test PUBLIC GTest::gtest_main mpcf_nopy ${CUDART_LIBRARY})

  if(ENABLE_COVERAGE)
    target_link_libraries(mpcf_nopy PRIVATE coverage_flags)
    target_link_libraries(mpcf_test PRIVATE coverage_flags)
  endif()

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
  
  target_link_libraries(mpcf_bench PUBLIC ${CPP_MODULE_NAME} ${CUDART_LIBRARY} ${Boost_LIBRARIES} Threads::Threads)
endif()

set_target_properties(${CPP_MODULE_NAME} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

if(MSVC)
  target_compile_options(${CPP_MODULE_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>")
else()
  #target_compile_options(${CPP_MODULE_NAME} PRIVATE -Wall -Wextra)
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

cmake_policy(SET CMP0177 NEW) # Normalize paths

#add_custom_command(
#        TARGET ${CPP_MODULE_NAME} POST_BUILD
#        COMMAND ${Python_EXECUTABLE} -m pybind11_stubgen ${CPP_MODULE_NAME}
#        --output-dir "${CMAKE_CURRENT_BINARY_DIR}"
#        COMMENT "Generating Python stubs for ${CPP_MODULE_NAME}..."
#)

#set(STUB_FILE "${CMAKE_CURRENT_BINARY_DIR}/${CPP_MODULE_NAME}.pyi")
#set_source_files_properties(${STUB_FILE} PROPERTIES GENERATED TRUE)
#file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/py.typed" "")

if (MINIMAL_MODULE_BUILD)
  message(STATUS "Python Executable: ${Python_EXECUTABLE}")

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "if True:
      import sysconfig
      print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE PYTHON_SITE_PRE
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  file(TO_CMAKE_PATH "${PYTHON_SITE_PRE}" PYTHON_SITE)

  SET(SITE_PKG_MASSPCF ${PYTHON_SITE}/masspcf)

  message(STATUS "Minimal module will be installed into ${SITE_PKG_MASSPCF}")

  file(GLOB PY_PKG_FILES "masspcf/*.py")
  #message(STATUS ${PY_PKG_FILES})
  install(
    DIRECTORY masspcf
    DESTINATION ${PYTHON_SITE}

    # Allow symlinking in CPP module for completions in IDE during development (__pycache__ should never be installed)
    REGEX "(__pycache__|\\.so$|\\.pyi$|py\\.typed$)" EXCLUDE
    )

  set(INSTALL_TARGET ${SITE_PKG_MASSPCF})

  # Copy on Windows, symlink otherwise
  if (WIN32)
    SET(LINK_TYPE copy_if_different)
  else()
    SET(LINK_TYPE create_symlink)
  endif()

  if(0)
    add_custom_command(
      TARGET ${CPP_MODULE_NAME}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${PYTHON_SITE}/masspcf/$<TARGET_FILE_NAME:${CPP_MODULE_NAME}>"
      "${PYTHON_SITE}/masspcf/py.typed"
      "${PYTHON_SITE}/masspcf/${CPP_MODULE_NAME}.pyi"
      "${CMAKE_CURRENT_SOURCE_DIR}/masspcf/"
      COMMENT "Syncing ${CPP_MODULE_NAME} to source-local directory"
      VERBATIM
    )
  endif()

  # Optional: Clean the symlink when running 'make clean'
  #set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${SYMLINK_NAME}")


else()
  set(INSTALL_TARGET masspcf)
endif()

install(TARGETS ${CPP_MODULE_NAME} DESTINATION ${INSTALL_TARGET})
#install(FILES ${STUB_FILE} DESTINATION ${INSTALL_TARGET})
#install(FILES "${CMAKE_CURRENT_BINARY_DIR}/py.typed" DESTINATION ${INSTALL_TARGET})

if (SKBUILD)
  # Install Python source files alongside compiled module
  install(
          DIRECTORY masspcf/
          DESTINATION ${INSTALL_TARGET}
          FILES_MATCHING PATTERN "*.py"
          PATTERN "__pycache__" EXCLUDE
  )
endif()

message(STATUS "!!! Stub file: ${STUB_FILE}")
message(STATUS "!!! Install target: ${INSTALL_TARGET}")

message(STATUS "!!! C compiler:        ${CMAKE_C_COMPILER}")
message(STATUS "!!! C++ compiler:      ${CMAKE_CXX_COMPILER}")
message(STATUS "!!! Compiler ID:       ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "!!! Python_ROOT_DIR:   ${Python_ROOT_DIR}")
message(STATUS "!!! Python_EXECUTABLE: ${Python_EXECUTABLE}")
