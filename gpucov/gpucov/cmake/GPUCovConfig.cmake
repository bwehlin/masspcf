# GPUCovConfig.cmake — CMake integration for GPUCov
#
# Provides:
#   gpucov_instrument_target(<target>
#       FILES <file1> [<file2> ...]
#       [SOURCE_ROOT <dir>]
#       [INCLUDE_PATHS <path1> [<path2> ...]]
#       [EXTRA_ARGS <arg1> [<arg2> ...]]
#   )
#
# This function:
#   1. Runs the gpucov instrumenter on the listed FILES
#   2. Creates a shadow directory with instrumented copies
#   3. Symlinks non-instrumented siblings so relative #includes resolve
#   4. Replaces .cu sources in the target with instrumented copies
#   5. Prepends shadow include paths so instrumented headers take precedence
#   6. Defines GPUCOV_ENABLED=1 and GPUCOV_MAX_COUNTERS=N on the target
#
# Injected code is guarded with #ifdef __CUDACC__ so files compiled by both
# the host compiler and NVCC work automatically — no special configuration
# is needed for dual-compilation headers.
#
# The target is only modified if the GPUCOV_ENABLE cache variable is ON.
# Set it via: cmake -DGPUCOV_ENABLE=ON  or  environment ENABLE_CUDA_COVERAGE=1
#
# Prerequisites:
#   - Python interpreter found (Python_EXECUTABLE or Python3_EXECUTABLE)
#   - gpucov pip package installed in that interpreter
#
# Example:
#   find_package(GPUCov)
#   if (GPUCOV_FOUND)
#       gpucov_instrument_target(my_cuda_lib
#           FILES
#               src/cuda/kernels.cu
#               include/cuda/kernels.cuh
#       )
#   endif()

# Determine enable state: environment variable overrides cache, then cache, then OFF.
# Environment takes precedence so that CI can toggle coverage without clearing the cache.
if ((DEFINED ENV{ENABLE_CUDA_COVERAGE}) AND (NOT "$ENV{ENABLE_CUDA_COVERAGE}" STREQUAL "0"))
    set(GPUCOV_ENABLE ON CACHE BOOL "Enable GPUCov CUDA device code coverage" FORCE)
elseif (NOT DEFINED GPUCOV_ENABLE)
    set(GPUCOV_ENABLE OFF CACHE BOOL "Enable GPUCov CUDA device code coverage")
endif()

# Find a Python interpreter
if (NOT DEFINED _GPUCOV_PYTHON)
    if (DEFINED Python_EXECUTABLE)
        set(_GPUCOV_PYTHON "${Python_EXECUTABLE}")
    elseif (DEFINED Python3_EXECUTABLE)
        set(_GPUCOV_PYTHON "${Python3_EXECUTABLE}")
    else()
        find_package(Python3 QUIET COMPONENTS Interpreter)
        if (Python3_FOUND)
            set(_GPUCOV_PYTHON "${Python3_EXECUTABLE}")
        else()
            find_program(_GPUCOV_PYTHON NAMES python3 python)
        endif()
    endif()
endif()

# Check gpucov is importable
if (_GPUCOV_PYTHON)
    # Run from CMAKE_BINARY_DIR to avoid shadowing the installed package
    # with a same-named project directory in the source tree.
    execute_process(
        COMMAND "${_GPUCOV_PYTHON}" -c "import gpucov; print(gpucov.__version__)"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        OUTPUT_VARIABLE _GPUCOV_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _GPUCOV_IMPORT_RESULT
    )
    if (_GPUCOV_IMPORT_RESULT EQUAL 0)
        set(GPUCOV_FOUND TRUE)
        set(GPUCOV_VERSION "${_GPUCOV_VERSION}")
        message(STATUS "Found GPUCov ${GPUCOV_VERSION} (${_GPUCOV_PYTHON})")
    else()
        set(GPUCOV_FOUND FALSE)
        if (GPUCov_FIND_REQUIRED)
            message(FATAL_ERROR "GPUCov: 'import gpucov' failed with ${_GPUCOV_PYTHON}. Install with: pip install gpucov")
        endif()
    endif()
else()
    set(GPUCOV_FOUND FALSE)
    if (GPUCov_FIND_REQUIRED)
        message(FATAL_ERROR "GPUCov: no Python interpreter found")
    endif()
endif()

function(gpucov_instrument_target TARGET)
    if (NOT GPUCOV_ENABLE)
        return()
    endif()
    if (NOT GPUCOV_FOUND)
        message(WARNING "gpucov_instrument_target: GPUCov not found, skipping instrumentation")
        return()
    endif()

    cmake_parse_arguments(
        _GC                                        # prefix
        ""                                         # options (flags)
        "SOURCE_ROOT"                              # single-value keywords
        "FILES;INCLUDE_PATHS;EXTRA_ARGS"  # multi-value keywords
        ${ARGN}
    )

    if (NOT _GC_FILES)
        message(FATAL_ERROR "gpucov_instrument_target: FILES is required")
    endif()

    # Default SOURCE_ROOT to CMAKE_SOURCE_DIR
    if (NOT _GC_SOURCE_ROOT)
        set(_GC_SOURCE_ROOT "${CMAKE_SOURCE_DIR}")
    endif()

    set(_SHADOW_DIR "${CMAKE_BINARY_DIR}/_gpucov_${TARGET}")

    # Build the instrumenter command
    set(_CMD "${_GPUCOV_PYTHON}" -m gpucov instrument
        --source-root "${_GC_SOURCE_ROOT}"
        --output-dir "${_SHADOW_DIR}"
    )

    if (_GC_INCLUDE_PATHS)
        foreach(_inc ${_GC_INCLUDE_PATHS})
            list(APPEND _CMD -I "${_inc}")
        endforeach()
    endif()

    if (_GC_EXTRA_ARGS)
        list(APPEND _CMD --extra-args)
        foreach(_arg ${_GC_EXTRA_ARGS})
            list(APPEND _CMD "${_arg}")
        endforeach()
    endif()

    list(APPEND _CMD --files)
    foreach(_file ${_GC_FILES})
        # Resolve to absolute path
        if (NOT IS_ABSOLUTE "${_file}")
            set(_file "${_GC_SOURCE_ROOT}/${_file}")
        endif()
        list(APPEND _CMD "${_file}")
    endforeach()

    # Run the instrumenter
    execute_process(
        COMMAND ${_CMD}
        WORKING_DIRECTORY "${_GC_SOURCE_ROOT}"
        RESULT_VARIABLE _GC_RESULT
        OUTPUT_VARIABLE _GC_OUTPUT
        ERROR_VARIABLE  _GC_ERROR
    )

    if (NOT _GC_RESULT EQUAL 0)
        message(WARNING "gpucov_instrument_target(${TARGET}): instrumentation failed:\n${_GC_OUTPUT}\n${_GC_ERROR}")
        return()
    endif()

    message(STATUS "GPUCov [${TARGET}]: ${_GC_OUTPUT}")

    # Read the mapping to get counter count
    set(_MAPPING_FILE "${_SHADOW_DIR}/mapping.json")
    if (NOT EXISTS "${_MAPPING_FILE}")
        message(WARNING "gpucov_instrument_target(${TARGET}): mapping.json not found")
        return()
    endif()

    file(READ "${_MAPPING_FILE}" _MAPPING_JSON)
    string(JSON _NUM_COUNTERS GET "${_MAPPING_JSON}" "num_counters")

    # Replace .cu sources with instrumented copies
    foreach(_file ${_GC_FILES})
        if (NOT IS_ABSOLUTE "${_file}")
            set(_file "${_GC_SOURCE_ROOT}/${_file}")
        endif()
        if ("${_file}" MATCHES "\\.cu$")
            file(RELATIVE_PATH _rel "${_GC_SOURCE_ROOT}" "${_file}")
            set(_shadow_cu "${_SHADOW_DIR}/${_rel}")
            if (EXISTS "${_shadow_cu}")
                target_sources(${TARGET} PRIVATE "${_shadow_cu}")
                # Remove original from target sources
                file(RELATIVE_PATH _rel_to_build "${CMAKE_SOURCE_DIR}" "${_file}")
                get_target_property(_sources ${TARGET} SOURCES)
                list(REMOVE_ITEM _sources "${_rel_to_build}")
                list(REMOVE_ITEM _sources "${_file}")
                set_target_properties(${TARGET} PROPERTIES SOURCES "${_sources}")
            endif()
        endif()
    endforeach()

    # Symlink non-instrumented files into the shadow tree so relative
    # includes from instrumented headers still resolve.
    # Walk every include path on the target; for each, symlink its
    # entire file tree into the matching shadow subtree.
    get_target_property(_inc_dirs ${TARGET} INCLUDE_DIRECTORIES)
    if (_inc_dirs)
        foreach(_inc_dir ${_inc_dirs})
            # Only process include dirs that are under SOURCE_ROOT
            file(RELATIVE_PATH _rel_inc "${_GC_SOURCE_ROOT}" "${_inc_dir}")
            if ("${_rel_inc}" MATCHES "^\\.\\.")
                continue()
            endif()
            file(GLOB_RECURSE _orig_files "${_inc_dir}/*")
            foreach(_orig ${_orig_files})
                file(RELATIVE_PATH _rel "${_GC_SOURCE_ROOT}" "${_orig}")
                set(_shadow "${_SHADOW_DIR}/${_rel}")
                if (NOT EXISTS "${_shadow}")
                    get_filename_component(_shadow_dir "${_shadow}" DIRECTORY)
                    file(MAKE_DIRECTORY "${_shadow_dir}")
                    file(CREATE_LINK "${_orig}" "${_shadow}" SYMBOLIC)
                endif()
            endforeach()
        endforeach()
    endif()

    # Prepend shadow include paths
    # Find which subdirectories of the shadow tree contain instrumented headers
    foreach(_file ${_GC_FILES})
        if (NOT IS_ABSOLUTE "${_file}")
            set(_file "${_GC_SOURCE_ROOT}/${_file}")
        endif()
        if ("${_file}" MATCHES "\\.cuh$")
            file(RELATIVE_PATH _rel "${_GC_SOURCE_ROOT}" "${_file}")
            get_filename_component(_rel_dir "${_rel}" DIRECTORY)
            # Walk up to find the include root (first path component)
            string(FIND "${_rel_dir}" "/" _slash_pos)
            if (_slash_pos GREATER -1)
                string(SUBSTRING "${_rel_dir}" 0 ${_slash_pos} _inc_root)
            else()
                set(_inc_root "${_rel_dir}")
            endif()
            list(APPEND _shadow_inc_roots "${_SHADOW_DIR}/${_inc_root}")
        endif()
    endforeach()

    if (_shadow_inc_roots)
        list(REMOVE_DUPLICATES _shadow_inc_roots)
        foreach(_sinc ${_shadow_inc_roots})
            target_include_directories(${TARGET} BEFORE PRIVATE "${_sinc}")
        endforeach()
    endif()

    # Add shadow root so gpucov_runtime.cuh can be found
    target_include_directories(${TARGET} BEFORE PRIVATE "${_SHADOW_DIR}")

    # Compile definitions
    target_compile_definitions(${TARGET} PRIVATE
        GPUCOV_ENABLED=1
        GPUCOV_MAX_COUNTERS=${_NUM_COUNTERS}
    )

    # Export the shadow dir and mapping path for the collect step
    set(GPUCOV_${TARGET}_SHADOW_DIR "${_SHADOW_DIR}" PARENT_SCOPE)
    set(GPUCOV_${TARGET}_MAPPING "${_MAPPING_FILE}" PARENT_SCOPE)
endfunction()
