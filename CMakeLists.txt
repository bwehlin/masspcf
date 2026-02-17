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

# Get the version from pyproject.toml
execute_process(
        COMMAND python3 -c "import tomllib; print(tomllib.load(open('${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml', 'rb'))['project']['version'])"
        OUTPUT_VARIABLE PROJ_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

#set(PROJ_VERSION "0.4.0")

message(STATUS "!!! Detected version ${PROJ_VERSION} from pyproject.toml")

project(${PROJ_NAME} VERSION ${PROJ_VERSION} LANGUAGES CXX ${LANG_CUDA})

message(STATUS "!!! PROJECT_NAME: ${PROJECT_NAME}, PROJECT_VERSION: ${PROJECT_VERSION} !!!")

set(CPP_MODULE_NAME "mpcf_cpp")

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

find_package(Python 3.10 REQUIRED COMPONENTS Development.Embed Development.Module Interpreter)

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
  algorithm.h
  algorithms/array_functional.h
  algorithms/iterate_rectangles.h
  algorithms/reduce.h
  algorithms/matrix_integrate.h
  algorithms/matrix_reduce.h
  algorithms/subdivide.h
  algorithms/apply_functional.h
  block_matrix_support.cuh)

SET(MPCF_SRC_FILES
  executor.cpp)

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
  py_make_from_serial_content.h py_make_from_serial_content.cpp
  pyarray_norms.h pyarray_norms.cpp
  py_reductions.h py_reductions.cpp
  py_tensor.h py_tensor.cpp
)

list(TRANSFORM MPCF_PY_SOURCES PREPEND "src/python/")


if (SKBUILD OR MINIMAL_MODULE_BUILD)
  python_add_library(${CPP_MODULE_NAME} MODULE ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES} WITH_SOABI)
else()
  python_add_library(${CPP_MODULE_NAME} SHARED ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES})
endif()

if (BUILD_WITH_CUDA)
  SET(CUDA_SEPARABLE_COMPILATION ON)
endif()

set(MPCF_PRIVATE_INCLUDES "include/mpcf")
set(MPCF_PUBLIC_INCLUDES "include/")

target_include_directories(${CPP_MODULE_NAME} PRIVATE ${MPCF_PRIVATE_INCLUDES})
target_include_directories(${CPP_MODULE_NAME} PUBLIC ${MPCF_PUBLIC_INCLUDES})

include_directories("3rd/xtensor/include" "3rd/xtl/include")
target_link_libraries(${CPP_MODULE_NAME} PRIVATE pybind11::headers)
target_compile_definitions(${CPP_MODULE_NAME} PRIVATE VERSION_INFO=${PROJECT_VERSION}) 

if (MSVC)
  # MSVC CUDA has some problems with picking up include directories in IntelliSense. We can solve this
  # by using VC++ Include Directories instead of on the compiler-level
  get_filename_component(taskflow_DIR "3rd/taskflow" REALPATH)
  set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$(VC_IncludePath);$(WindowsSDK_IncludePath);${taskflow_DIR};${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

if (BUILD_TESTER)
  set(BUILD_GMOCK OFF)
  set(INSTALL_GTEST OFF)
  set(gtest_force_shared_crt ON)

  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  add_subdirectory("3rd/googletest")

  #message(STATUS "-- GoogleTest version: ${GOOGLETEST_VERSION}")

  #add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
  enable_testing()
  add_executable(mpcf_test
    test/test_block_matrix_support.cu
    test/test_iterate_rectangles.cpp
    test/test_l1_dist.cu
    test/test_norms.cu
    test/test_parallel_reduce.cpp
    test/test_tensor_iteration.cpp
    test/test_special_tensors.cpp)

  target_include_directories(mpcf_test PRIVATE "include/")
  target_include_directories(mpcf_test SYSTEM PRIVATE "3rd/xtensor/include" "3rd/xtl/include")

  python_add_library(mpcf_nopy STATIC ${MPCF_LIB_SOURCES})
  target_include_directories(mpcf_nopy PRIVATE ${MPCF_PRIVATE_INCLUDES})
  target_include_directories(mpcf_nopy PUBLIC ${MPCF_PUBLIC_INCLUDES})

  target_link_libraries(mpcf_test PUBLIC GTest::gtest_main mpcf_nopy ${CUDART_LIBRARY})
  
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

add_custom_command(
        TARGET ${CPP_MODULE_NAME} POST_BUILD
        COMMAND ${Python_EXECUTABLE} -m pybind11_stubgen ${CPP_MODULE_NAME}
        --output-dir "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating Python stubs for ${CPP_MODULE_NAME}..."
)

set(STUB_FILE "${CMAKE_CURRENT_BINARY_DIR}/${CPP_MODULE_NAME}.pyi")
set_source_files_properties(${STUB_FILE} PROPERTIES GENERATED TRUE)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/py.typed" "")

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

  install(CODE "
    set(LINK_PATH \"${CMAKE_CURRENT_SOURCE_DIR}/masspcf/${CPP_MODULE_NAME}$<TARGET_FILE_SUFFIX:${CPP_MODULE_NAME}>\")
    set(TARGET_PATH \"${PYTHON_SITE}/masspcf/$<TARGET_FILE_NAME:${CPP_MODULE_NAME}>\")

    message(STATUS \"Creating symlink: \${LINK_PATH} -> \${TARGET_PATH}\")

    execute_process(
      COMMAND \${CMAKE_COMMAND} -E ${LINK_TYPE}
      \"\${TARGET_PATH}\"
      \"\${LINK_PATH}\"
    )
  ")

  install(CODE "
    set(LINK_PATH \"${CMAKE_CURRENT_SOURCE_DIR}/masspcf/py.typed\")
    set(TARGET_PATH \"${PYTHON_SITE}/masspcf/py.typed\")

    message(STATUS \"Creating symlink: \${LINK_PATH} -> \${TARGET_PATH}\")

    execute_process(
      COMMAND \${CMAKE_COMMAND} -E ${LINK_TYPE}
      \"\${TARGET_PATH}\"
      \"\${LINK_PATH}\"
    )
  ")

  install(CODE "
    set(LINK_PATH \"${CMAKE_CURRENT_SOURCE_DIR}/masspcf/${CPP_MODULE_NAME}.pyi\")
    set(TARGET_PATH \"${PYTHON_SITE}/masspcf/${CPP_MODULE_NAME}.pyi\")

    message(STATUS \"Creating symlink: \${LINK_PATH} -> \${TARGET_PATH}\")

    execute_process(
      COMMAND \${CMAKE_COMMAND} -E ${LINK_TYPE}
      \"\${TARGET_PATH}\"
      \"\${LINK_PATH}\"
    )
  ")


  # Optional: Clean the symlink when running 'make clean'
  #set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${SYMLINK_NAME}")


else()
  set(INSTALL_TARGET masspcf)
endif()

install(TARGETS ${CPP_MODULE_NAME} DESTINATION ${INSTALL_TARGET})
install(FILES ${STUB_FILE} DESTINATION ${INSTALL_TARGET})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/py.typed" DESTINATION ${INSTALL_TARGET})

message(STATUS "!!! Stub file: ${STUB_FILE}")
message(STATUS "!!! Install target: ${INSTALL_TARGET}")

message(STATUS "!!! C compiler:        ${CMAKE_C_COMPILER}")
message(STATUS "!!! C++ compiler:      ${CMAKE_CXX_COMPILER}")
message(STATUS "!!! Compiler ID:       ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "!!! Python_ROOT_DIR:   ${Python_ROOT_DIR}")
message(STATUS "!!! Python_EXECUTABLE: ${Python_EXECUTABLE}")
