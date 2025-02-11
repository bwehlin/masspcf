cmake_minimum_required(VERSION 3.16...3.29)

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

if (BUILD_WITH_CUDA)
  SET(LANG_CUDA CUDA)
  add_compile_definitions(BUILD_WITH_CUDA=1)
else()
  SET(LANG_CUDA )
endif()

if (WIN32)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
endif()
 
if (NOT SKBUILD)
  set(SKBUILD_PROJECT_NAME massivepcf)
  set(SKBUILD_PROJECT_VERSION "0.3.1")
endif()

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX ${LANG_CUDA})

if (SKBUILD)
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
else()
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Embed)
endif()

include_directories("3rd/taskflow")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


add_subdirectory(3rd/pybind11)
add_subdirectory(3rd/xtl)
add_subdirectory(3rd/xtensor)

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
  include/mpcf/random.h src/random.cpp
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
  src/python/pymodule.cu
  src/python/pyrandom.cpp
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

target_include_directories(mpcf_cpp SYSTEM PRIVATE "3rd/xtensor/include" "3rd/xtl/include")
target_link_libraries(mpcf_cpp PRIVATE pybind11::headers)
target_compile_definitions(mpcf_cpp PRIVATE VERSION_INFO=${PROJECT_VERSION}) 

if (MSVC)
  # MSVC CUDA has some problems with picking up include directories in IntelliSense. We can solve this
  # by using VC++ Include Directories instead of on the compiler-level
  get_filename_component(taskflow_DIR "3rd/taskflow" REALPATH)
  set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$(VC_IncludePath);$(WindowsSDK_IncludePath);${taskflow_DIR};${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

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


install(TARGETS mpcf_cpp DESTINATION masspcf)

if (BUILD_WITH_CUDA)

  find_package(CUDAToolkit REQUIRED)
  message("CUDART: ${CUDAToolkit_LIBRARY_DIR}")
  set(CUDART_LIB ${CUDAToolkit_LIBRARY_DIR}/libcudart.so)
  file(GLOB cudart_libs ${CUDAToolkit_LIBRARY_DIR}/libcudart.so*)
  message("cudart_libs: ${cudart_libs}")
  
  install(FILES ${cudart_libs} DESTINATION masspcf)

endif()
