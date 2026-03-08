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

if ((DEFINED ENV{MINIMAL_MODULE_BUILD}) AND (NOT $ENV{MINIMAL_MODULE_BUILD} MATCHES "0"))
  set(MINIMAL_MODULE_BUILD ON)
  message(STATUS "!!! Minimal module build !!!")
else ()
  message(STATUS "!!! Not Minimal module build !!!")
  set(MINIMAL_MODULE_BUILD OFF)
endif ()

if (DEFINED ENV{BUILD_WITH_CUDA})
  if ($ENV{BUILD_WITH_CUDA} MATCHES "0")
    set(BUILD_WITH_CUDA OFF)
  elseif (APPLE)
    set(BUILD_WITH_CUDA OFF)
  else ()
    set(BUILD_WITH_CUDA ON)
  endif ()
elseif (APPLE)
  set(BUILD_WITH_CUDA OFF)
else ()
  set(BUILD_WITH_CUDA ON)
endif ()

if ((DEFINED ENV{SKIP_STUBGEN}) AND (NOT $ENV{SKIP_STUBGEN} MATCHES "0"))
  set(SKIP_STUBGEN ON)
else ()
  set(SKIP_STUBGEN OFF)
endif ()

if ((DEFINED ENV{SKIP_BACK_COPY}) AND (NOT $ENV{SKIP_BACK_COPY} MATCHES "0"))
  set(SKIP_BACK_COPY ON)
else ()
  set(SKIP_BACK_COPY OFF)
endif ()

if (SKBUILD)
  set(BUILD_TESTER OFF)
  set(SKIP_STUBGEN ON) # Too many problems with stubgen and skbuild at the moment, we don't really need the stubs anyway in the release since the C++ bits are all internal...
else ()
  set(BUILD_TESTER ON)
endif ()

set(BUILD_BENCHMARKS OFF)

if (SKBUILD OR MINIMAL_MODULE_BUILD)
  set(BUILD_BENCHMARKS OFF)
endif ()

find_package(Python 3.10 REQUIRED COMPONENTS Development.Module Interpreter)

if (BUILD_WITH_CUDA)
  message(STATUS "!!! Building with CUDA !!!")

  # If CUDAToolkit_ROOT is not set, try to find a pip-installed CUDA toolkit
  if (NOT DEFINED CUDAToolkit_ROOT AND NOT DEFINED ENV{CUDAToolkit_ROOT})
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "if True:
        import importlib.util, pathlib
        for pkg in ('nvidia.cu13', 'nvidia.cu12', 'nvidia.cuda_nvcc', 'nvidia.cuda_runtime'):
            spec = importlib.util.find_spec(pkg)
            if spec and spec.submodule_search_locations:
                print(pathlib.Path(spec.submodule_search_locations[0]).resolve())
                break"
        OUTPUT_VARIABLE _PIP_CUDA_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if (_PIP_CUDA_ROOT)
      message(STATUS "!!! Found pip-installed CUDA toolkit: ${_PIP_CUDA_ROOT}")
      set(CUDAToolkit_ROOT "${_PIP_CUDA_ROOT}")
      set(CMAKE_CUDA_COMPILER "${_PIP_CUDA_ROOT}/bin/nvcc")

      # Pip-installed CUDA uses lib/ instead of lib64/; help nvcc and FindCUDAToolkit find the libraries
      if (EXISTS "${_PIP_CUDA_ROOT}/lib" AND NOT EXISTS "${_PIP_CUDA_ROOT}/lib64")
        # Pip CUDA ships only versioned .so files (e.g. libcudart.so.13).
        # Create unversioned symlinks in the build dir so the linker can
        # resolve -lcudart etc., without modifying the pip installation.
        set(_PIP_CUDA_LINK_DIR "${CMAKE_CURRENT_BINARY_DIR}/_pip_cuda_links")
        file(MAKE_DIRECTORY "${_PIP_CUDA_LINK_DIR}")
        file(GLOB _PIP_CUDA_VERSIONED_LIBS "${_PIP_CUDA_ROOT}/lib/lib*.so.*")
        foreach(_VLIB IN LISTS _PIP_CUDA_VERSIONED_LIBS)
          get_filename_component(_VLIB_NAME "${_VLIB}" NAME)
          string(REGEX REPLACE "\\.so\\..*" ".so" _UNVERSIONED_NAME "${_VLIB_NAME}")
          if (NOT EXISTS "${_PIP_CUDA_LINK_DIR}/${_UNVERSIONED_NAME}")
            file(CREATE_LINK "${_VLIB}" "${_PIP_CUDA_LINK_DIR}/${_UNVERSIONED_NAME}" SYMBOLIC)
          endif ()
        endforeach()

        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -L${_PIP_CUDA_ROOT}/lib -L${_PIP_CUDA_LINK_DIR}")
        link_directories("${_PIP_CUDA_LINK_DIR}" "${_PIP_CUDA_ROOT}/lib")
      endif ()
    endif ()
  endif ()

  SET(LANG_CUDA CUDA)

  if (MINIMAL_MODULE_BUILD)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    # This matches what torch.cuda.get_gencode_flags() returns for PyTorch 2.10.0. If it's good enough for them, it's
    # probably good enough for us. We may need to update this list sometime in the future as GPUs evolve.
    set(CMAKE_CUDA_ARCHITECTURES
      75
      80
      86
      90
      100
      120
    )
  endif()
else ()
  message(STATUS "!!! Building without CUDA !!!")

  SET(LANG_CUDA)
endif ()

if (SKBUILD OR MINIMAL_MODULE_BUILD)
  set(PROJ_NAME masspcf)
else ()
  set(PROJ_NAME massivepcf)
endif ()

# Read version info out of pyproject.toml
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" TOML_CONTENT)

# 1. Find the [project] section
string(FIND "${TOML_CONTENT}" "[project]" PROJECT_SECTION_START)

# 2. Slice the string from [project] onwards to avoid [build-system] versions
string(SUBSTRING "${TOML_CONTENT}" ${PROJECT_SECTION_START} -1 PROJECT_BODY)

# 3. Extract the version string
if (PROJECT_BODY MATCHES [[version[ \t]*=[ \t]*"([^"]+)"]])
  set(PROJ_VERSION_FULL "${CMAKE_MATCH_1}")

  # Extract only the numeric part for CMake's project() command
  string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" PROJ_VERSION_NUMERIC "${PROJ_VERSION_FULL}")

  message(STATUS "!!! Detected Full Version: ${PROJ_VERSION_FULL}")
  message(STATUS "!!! Numeric Version for CMake: ${PROJ_VERSION_NUMERIC}")

  set(PROJ_VERSION "${PROJ_VERSION_NUMERIC}")
else ()
  message(FATAL "Could not detect version number from pyproject.toml!")
endif ()

project(${PROJ_NAME} VERSION ${PROJ_VERSION_NUMERIC} LANGUAGES CXX ${LANG_CUDA})

if (BUILD_WITH_CUDA)
  if (_PIP_CUDA_ROOT AND EXISTS "${_PIP_CUDA_ROOT}/lib" AND NOT EXISTS "${_PIP_CUDA_ROOT}/lib64")
    # Pip-installed CUDA uses lib/ instead of lib64/ and ships only versioned
    # .so files (e.g. libcudart.so.13 instead of libcudart.so).
    # FindCUDAToolkit can't find these, so we locate them ourselves.
    file(GLOB _PIP_CUDART "${_PIP_CUDA_ROOT}/lib/libcudart.so*")
    if (_PIP_CUDART)
      list(GET _PIP_CUDART 0 _PIP_CUDART_LIB)
      set(CUDA_CUDART "${_PIP_CUDART_LIB}" CACHE FILEPATH "CUDA runtime library" FORCE)
    endif ()
  endif ()
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
endif ()

if ((DEFINED ENV{ENABLE_COVERAGE}) AND (NOT $ENV{ENABLE_COVERAGE} MATCHES "0"))
  set(ENABLE_COVERAGE ON)
else ()
  set(ENABLE_COVERAGE OFF)
endif ()

message(STATUS "!!! PROJECT_NAME: ${PROJECT_NAME}, PROJECT_VERSION: ${PROJECT_VERSION} !!!")

set(CPP_MODULE_NAME_CPU "_mpcf_cpp_cpu")
set(CPP_MODULE_NAMES ${CPP_MODULE_NAME_CPU})

if (BUILD_WITH_CUDA)
  set(CPP_MODULE_NAME_CUDA "_mpcf_cpp_cuda")
  list(APPEND CPP_MODULE_NAMES ${CPP_MODULE_NAME_CUDA})
endif ()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CXX)

if (WIN32)
  if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
  else ()
    add_compile_options(-Wa,-mbig-obj)
  endif ()
endif ()

include_directories("3rd/taskflow")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(3rd/pybind11)

# Adapted from https://community.gigperformer.com/t/getting-cmake-variables-from-c/17711
string(TIMESTAMP TODAY "%d/%m/%Y")
set(PROJECT_BUILD_DATE ${TODAY})
configure_file(version.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/version.cpp @ONLY)

SET(MPCF_LIB_SOURCES_CUDA
    include/mpcf/cuda/cuda_matrix_integrate.cuh
    include/mpcf/cuda/cuda_util.cuh
    include/mpcf/cuda/cuda_device_array.cuh
    include/mpcf/cuda/cuda_functional_support.cuh
    include/mpcf/cuda/cuda_matrix_integrate_structs.cuh
)

SET(MPCF_HEADER_FILES
    config.h
    version.h
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
    point_cloud.h
    io.h
    io/io_stream.h
    io/io_stream_base.h
    io/barcode_io.h
    io/pcf_io.h
    io/point_io.h
    io/point_cloud_io.h
    io/tensor_io.h
    algorithm.h
    algorithms/iterate_rectangles.h
    algorithms/reduce.h
    algorithms/matrix_integrate.h
    algorithms/matrix_reduce.h
    algorithms/subdivide.h
    algorithms/apply_functional.h
    block_matrix_support.cuh
    persistence/barcode.h
    persistence/compute_persistence.h
    persistence/persistence_pair.h
    persistence/stable_rank.h
    persistence/ripser/ripser.h)

SET(MPCF_SRC_FILES
    executor.cpp
    io_stream.cpp)

list(TRANSFORM MPCF_HEADER_FILES PREPEND "include/mpcf/")
list(TRANSFORM MPCF_SRC_FILES PREPEND "src/")

SET(MPCF_LIB_SOURCES
    ${MPCF_HEADER_FILES}
    ${MPCF_SRC_FILES}
)

SET(MPCF_PY_PERSISTENCE_SOURCES
    py_barcode.h py_barcode.cpp
    py_persistence_pair.h py_persistence_pair.cpp
    py_ripser.h py_ripser.cpp
    py_stable_rank.h py_stable_rank.cpp
    pymodule_persistence.h pymodule_persistence.cpp
)

list(TRANSFORM MPCF_PY_PERSISTENCE_SOURCES PREPEND "persistence/")

SET(MPCF_PY_SOURCES
    pybind.h
    pymodule.cpp
    pyrandom.cpp
    pypcf_support.h
    py_np_support.h
    py_np_tensor_convert.h py_np_tensor_convert.cpp
    py_make_from_serial_content.h py_make_from_serial_content.cpp
    py_distance.h py_distance.cpp
    py_norms.h py_norms.cpp
    py_reductions.h py_reductions.cpp
    py_tensor.h py_tensor.cpp
    py_io.h py_io.cpp
    ${MPCF_PY_PERSISTENCE_SOURCES}
)

list(TRANSFORM MPCF_PY_SOURCES PREPEND "src/python/")

SET(MPCF_VERSION_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)

# Standalone GPU detection module (no CUDA dependency)
pybind11_add_module(_gpu_detect MODULE src/gpu_detect/gpu_detect.cpp)
target_link_libraries(_gpu_detect PRIVATE pybind11::headers)
if (APPLE)
  target_link_libraries(_gpu_detect PRIVATE "-framework IOKit" "-framework CoreFoundation")
elseif (WIN32)
  target_link_libraries(_gpu_detect PRIVATE dxgi)
endif ()

set(MPCF_PRIVATE_INCLUDES "include/mpcf")
set(MPCF_PUBLIC_INCLUDES "include/")

if (BUILD_WITH_CUDA)
  # Build only the .cu files with NVCC, in a separate static library.
  # Everything else stays compiled by the host C++ compiler.
  add_library(mpcf_cuda STATIC
      src/cuda/cuda_matrix_integrate.cu
  )
  target_include_directories(mpcf_cuda PRIVATE ${MPCF_PRIVATE_INCLUDES})
  target_include_directories(mpcf_cuda PUBLIC ${MPCF_PUBLIC_INCLUDES})
  target_include_directories(mpcf_cuda PRIVATE "3rd/taskflow")
  target_include_directories(mpcf_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_compile_definitions(mpcf_cuda PRIVATE BUILD_WITH_CUDA=1)

  set_target_properties(mpcf_cuda PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      POSITION_INDEPENDENT_CODE ON
  )

  target_compile_options(mpcf_cuda PRIVATE
      --expt-relaxed-constexpr
      -Xcudafe=--diag_suppress=partial_override
      -Xcudafe=--diag_suppress=virtual_function_decl_hidden
      -Xcudafe=--diag_suppress=initialization_not_reachable
  )
endif ()

# Create both CPU and (optionally) CUDA Python modules
foreach(MOD_NAME IN LISTS CPP_MODULE_NAMES)
  if (MOD_NAME STREQUAL CPP_MODULE_NAME_CUDA)
    pybind11_add_module(${MOD_NAME} MODULE ${MPCF_LIB_SOURCES} ${MPCF_LIB_SOURCES_CUDA} ${MPCF_PY_SOURCES} ${MPCF_VERSION_SOURCE})
    target_compile_definitions(${MOD_NAME} PRIVATE BUILD_WITH_CUDA=1)
    target_link_libraries(${MOD_NAME} PRIVATE mpcf_cuda ${CUDAToolkit_CUDART_LIBRARY})
  else ()
    pybind11_add_module(${MOD_NAME} MODULE ${MPCF_LIB_SOURCES} ${MPCF_PY_SOURCES} ${MPCF_VERSION_SOURCE})
  endif ()

  target_compile_definitions(${MOD_NAME} PRIVATE MPCF_MODULE_NAME=${MOD_NAME})
  target_include_directories(${MOD_NAME} PRIVATE ${MPCF_PRIVATE_INCLUDES})
  target_include_directories(${MOD_NAME} PUBLIC ${MPCF_PUBLIC_INCLUDES})
  target_link_libraries(${MOD_NAME} PRIVATE pybind11::headers)
  target_compile_definitions(${MOD_NAME} PRIVATE VERSION_INFO=${PROJECT_VERSION_FULL})
endforeach()

# Convenience target so CLion (and plain cmake --build) can build all Python modules at once
add_custom_target(_mpcf_cpp DEPENDS ${CPP_MODULE_NAMES})

if (MSVC)
  # MSVC CUDA has some problems with picking up include directories in IntelliSense. We can solve this
  # by using VC++ Include Directories instead of on the compiler-level
  get_filename_component(taskflow_DIR "3rd/taskflow" REALPATH)
  set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$(VC_IncludePath);$(WindowsSDK_IncludePath);${taskflow_DIR};${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif ()

# ============================================================
# Coverage configuration (GCC or Clang/LLVM)
# ============================================================
if (ENABLE_COVERAGE)
  # Create the interface library only once even if this block is re-entered.
  if (NOT TARGET coverage_flags)
    add_library(coverage_flags INTERFACE)
  endif ()

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "!!! Coverage ENABLED (LLVM/Clang) !!!")
    # Disable optimization for accurate coverage; use LLVM source-based coverage.
    target_compile_options(coverage_flags INTERFACE
      -O0 -g
      -fprofile-instr-generate
      -fcoverage-mapping
    )
    # With Clang/LLVM, pass profiling to the linker too (do NOT link gcov).
    target_link_options(coverage_flags INTERFACE -fprofile-instr-generate)

  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "!!! Coverage ENABLED (GCC/gcov) !!!")
    # Existing GCC/gcov-style coverage (unchanged).
    target_compile_options(coverage_flags INTERFACE
      -O0 -g
      --coverage # Shorthand for arcs/test-coverage
      -fprofile-abs-path # Essential for finding headers in GHA
      -fno-inline
      -fno-inline-small-functions
      -fno-default-inline
    )
    target_link_options(coverage_flags INTERFACE --coverage)
    target_link_libraries(coverage_flags INTERFACE gcov)

  else ()
    message(FATAL_ERROR "Coverage only supported with GCC or Clang/LLVM")
  endif ()
else ()
  message(STATUS "!!! Coverage DISABLED !!!")
endif ()

if (BUILD_TESTER)
  # On GitHub Actions there is no GPU, so build the tester CPU-only even when
  # the Python modules are built with CUDA. CUDA tests are excluded in this case.
  if (BUILD_WITH_CUDA AND NOT DEFINED ENV{GITHUB_ACTIONS})
    set(BUILD_CUDA_TESTER ON)
  else ()
    set(BUILD_CUDA_TESTER OFF)
    message(STATUS "!!! Tester built without CUDA (no GPU on this runner) !!!")
  endif ()

  set(BUILD_GMOCK OFF)
  set(INSTALL_GTEST OFF)
  set(gtest_force_shared_crt ON)

  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  add_subdirectory("3rd/googletest")

  enable_testing()

  set(MPCF_TEST_SOURCES_COMMON
      test/test_iterate_rectangles.cpp
      test/test_parallel_reduce.cpp
      test/test_tensor.cpp
      test/test_tensor_iteration.cpp
      test/test_special_tensors.cpp
      test/test_io_stream.cpp
      test/test_io_readwrite.cpp
      test/test_pcf_algebra_and_norms.cpp
      test/test_tensor_io_core.cpp
      test/test_barcode_and_stable_rank.cpp
  )

  if (BUILD_CUDA_TESTER)
    add_executable(mpcf_test
        test/test_block_matrix_support.cu
        test/test_l1_dist.cu
        test/test_norms.cu
        ${MPCF_TEST_SOURCES_COMMON}
    )
  else ()
    add_executable(mpcf_test ${MPCF_TEST_SOURCES_COMMON})
  endif ()

  target_include_directories(mpcf_test PRIVATE "include/")

  if (BUILD_CUDA_TESTER)
    add_library(mpcf_nopy STATIC ${MPCF_LIB_SOURCES} ${MPCF_LIB_SOURCES_CUDA} ${MPCF_VERSION_SOURCE})
    target_compile_definitions(mpcf_nopy PUBLIC BUILD_WITH_CUDA=1)
    target_link_libraries(mpcf_nopy PRIVATE mpcf_cuda)
  else ()
    add_library(mpcf_nopy STATIC ${MPCF_LIB_SOURCES} ${MPCF_VERSION_SOURCE})
  endif ()
  target_include_directories(mpcf_nopy PRIVATE ${MPCF_PRIVATE_INCLUDES})
  target_include_directories(mpcf_nopy PUBLIC ${MPCF_PUBLIC_INCLUDES})

  if (BUILD_CUDA_TESTER)
    set_target_properties(mpcf_nopy PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared  # Must match the Python module's runtime
    )
    set_target_properties(mpcf_test PROPERTIES
        CUDA_RUNTIME_LIBRARY Static
    )
  endif ()

  target_link_libraries(mpcf_test PUBLIC GTest::gtest_main mpcf_nopy)

  if (BUILD_CUDA_TESTER)
    target_link_libraries(mpcf_test PUBLIC mpcf_cuda CUDA::cudart_static)
  endif ()

  if (ENABLE_COVERAGE)
    target_link_libraries(mpcf_nopy PRIVATE coverage_flags)
    target_link_libraries(mpcf_test PRIVATE coverage_flags)
    foreach(MOD_NAME IN LISTS CPP_MODULE_NAMES)
      target_link_libraries(${MOD_NAME} PRIVATE coverage_flags)
    endforeach()
  endif ()

  if (WIN32)
    get_filename_component(python_dir "${Python_EXECUTABLE}" DIRECTORY)
    set_target_properties(mpcf_test PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=${python_dir}")
  endif ()


endif ()

if (BUILD_BENCHMARKS)
  set(Boost_USE_STATIC_LIBS ON)
  set(Boost_USE_MULTITHREADED ON)

  find_package(Boost COMPONENTS program_options system REQUIRED)
  find_package(Threads REQUIRED)

  add_executable(mpcf_bench
      benchmarking/bench.cpp
      benchmarking/benchmark.h
      benchmarking/bench_reduction.h benchmarking/bench_reduction.cpp
  )

  set_target_properties(mpcf_bench PROPERTIES
      CUDA_RUNTIME_LIBRARY Static
  )

  target_link_libraries(mpcf_bench PUBLIC ${CPP_MODULE_NAME_CPU} ${Boost_LIBRARIES} Threads::Threads)

  if (BUILD_WITH_CUDA)
    target_link_libraries(mpcf_bench PUBLIC ${CPP_MODULE_NAME_CUDA} CUDA::cudart_static)
  endif ()
endif ()

foreach(MOD_NAME IN LISTS CPP_MODULE_NAMES)
  if (MSVC)
    target_compile_options(${MOD_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>")
  endif ()
endforeach()

if (NOT MSVC)
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror=return-type")
endif ()

if (POLICY CMP0177)
  cmake_policy(SET CMP0177 NEW) # Normalize paths
endif ()

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
  set(INSTALL_TARGET ${SITE_PKG_MASSPCF})

  message(STATUS "Minimal module will be installed into ${SITE_PKG_MASSPCF}")
else ()
  set(INSTALL_TARGET masspcf)
endif ()

install(TARGETS ${CPP_MODULE_NAMES} _gpu_detect DESTINATION ${INSTALL_TARGET})

if (LINUX OR MINIMAL_MODULE_BUILD AND (NOT SKIP_STUBGEN)) # Stubgen does not seem happy on CI
  foreach(MOD_NAME IN LISTS CPP_MODULE_NAMES)
    add_custom_command(
        TARGET ${MOD_NAME}
        POST_BUILD
        COMMAND ${Python_EXECUTABLE} -m pybind11_stubgen ${MOD_NAME}
        --output-dir "${CMAKE_CURRENT_BINARY_DIR}/stubs"
        --ignore-invalid-expressions comment
        COMMENT "Generating Python stubs for ${MOD_NAME}..."
    )

    install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/stubs/${MOD_NAME}/"
        DESTINATION "${INSTALL_TARGET}/${MOD_NAME}"
        FILES_MATCHING
        PATTERN "*.pyi"
    )
  endforeach()

  add_custom_command(
      TARGET ${CPP_MODULE_NAME_CPU}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E touch
      ${CMAKE_CURRENT_BINARY_DIR}/py.typed
      COMMENT "Creating py.typed"
  )

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/py.typed" "")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/py.typed" DESTINATION ${INSTALL_TARGET})
endif ()

if (SKBUILD)
  # Install Python source files alongside compiled module
  install(
      DIRECTORY masspcf/
      DESTINATION ${INSTALL_TARGET}
      FILES_MATCHING PATTERN "*.py"
      PATTERN "__pycache__" EXCLUDE
  )

elseif (MINIMAL_MODULE_BUILD)


  file(GLOB PY_PKG_FILES "masspcf/*.py")
  #message(STATUS ${PY_PKG_FILES})
  install(
      DIRECTORY masspcf
      DESTINATION ${PYTHON_SITE}

      # Allow symlinking in CPP module for completions in IDE during development (__pycache__ should never be installed)
      REGEX "(__pycache__|\\.so$|\\.pyi$|py\\.typed|_mpcf_cpp_cpu$|_mpcf_cpp_cuda$)" EXCLUDE
  )


  if (NOT SKIP_BACK_COPY)

    # Copy on Windows, symlink otherwise
    if (WIN32)
      SET(LINK_TYPE copy_if_different)
    else ()
      SET(LINK_TYPE create_symlink)
    endif ()

    foreach(MOD_NAME IN LISTS CPP_MODULE_NAMES)
      install(CODE "
      message(STATUS \"Syncing built extension ${MOD_NAME} back to source directory\")

      set(SRC_DIR \"${CMAKE_CURRENT_SOURCE_DIR}/masspcf\")

      # Extension file
      set(EXT_FILE \"$<TARGET_FILE:${MOD_NAME}>\")
      get_filename_component(EXT_NAME \"\${EXT_FILE}\" NAME)
      set(EXT_DEST \"\${SRC_DIR}/\${EXT_NAME}\")

      if(UNIX)
        file(CREATE_LINK \"\${EXT_FILE}\" \"\${EXT_DEST}\" SYMBOLIC)
      else()
        file(COPY \"\${EXT_FILE}\" DESTINATION \"\${SRC_DIR}\")
      endif()

      if (NOT SKIP_STUBGEN)

        # Stub file
        set(STUB_SRC_DIR \"${CMAKE_CURRENT_BINARY_DIR}/stubs/${MOD_NAME}\")
        set(STUB_DEST_DIR \"\${SRC_DIR}/${MOD_NAME}\")

        if(UNIX)
          # Check if the destination already exists (as a folder or a symlink)
          if(NOT EXISTS \"\${STUB_DEST_DIR}\")
            message(STATUS \"Creating initial symlink for stubs: \${STUB_DEST_DIR}\")
            file(CREATE_LINK \"\${STUB_SRC_DIR}\" \"\${STUB_DEST_DIR}\" SYMBOLIC)
          else()
            # Optional: Check if it's pointing to the WRONG place (e.g., after a build dir change)
            file(READ_SYMLINK \"\${STUB_DEST_DIR}\" CURRENT_LINK_TARGET)
            if(NOT \"\${CURRENT_LINK_TARGET}\" STREQUAL \"\${STUB_SRC_DIR}\")
              message(STATUS \"Updating stale symlink for stubs...\")
              file(REMOVE_RECURSE \"\${STUB_DEST_DIR}\")
              file(CREATE_LINK \"\${STUB_SRC_DIR}\" \"\${STUB_DEST_DIR}\" SYMBOLIC)
            else()
              message(STATUS \"Symlink for stubs already exists and is correct.\")
            endif()
          endif()
        endif()

        # py.typed (small file — copy everywhere)
        file(COPY
          \"${CMAKE_CURRENT_BINARY_DIR}/py.typed\"
          DESTINATION \"\${SRC_DIR}\")
      endif()
      ")
    endforeach()

    # Optional: Clean the symlink when running 'make clean'
    #set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${SYMLINK_NAME}")
  endif ()

endif ()

# ============================================================
# Valgrind / debug info configuration (Linux only)
# ============================================================

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND BUILD_TESTER)
  add_library(valgrind_flags INTERFACE)

  target_compile_options(valgrind_flags INTERFACE $<$<CONFIG:Debug>:-fno-pie>)
  target_link_options(valgrind_flags INTERFACE $<$<CONFIG:Debug>:-no-pie>)

  find_program(LSB_RELEASE lsb_release)
  if(LSB_RELEASE)
    execute_process(
      COMMAND ${LSB_RELEASE} -rs
      OUTPUT_VARIABLE DISTRO_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(DISTRO_VERSION VERSION_LESS "24.04")
      message(STATUS "Ubuntu < 24.04 detected, forcing -gdwarf-4 for valgrind compatibility")
      target_compile_options(valgrind_flags INTERFACE $<$<CONFIG:Debug>:-gdwarf-4>)
    endif()
  endif()

  target_link_libraries(mpcf_nopy PRIVATE valgrind_flags)
  target_link_libraries(mpcf_test PRIVATE valgrind_flags)
endif()

message(STATUS "!!! Stub file: ${STUB_FILE}")
message(STATUS "!!! Install target: ${INSTALL_TARGET}")

message(STATUS "!!! C compiler:        ${CMAKE_C_COMPILER}")
message(STATUS "!!! C++ compiler:      ${CMAKE_CXX_COMPILER}")
message(STATUS "!!! Compiler ID:       ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "!!! Python_ROOT_DIR:   ${Python_ROOT_DIR}")
message(STATUS "!!! Python_EXECUTABLE: ${Python_EXECUTABLE}")
