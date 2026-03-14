# clean_masspcf.cmake — invoked by the clean_masspcf target
# Expects: -DSRC_DIR=... -DSITE_DIR=...

# 1. Remove symlinks (and stale .so copies on Windows) from the source tree
file(GLOB _SRC_SO "${SRC_DIR}/*.so" "${SRC_DIR}/*.pyd")
file(GLOB _SRC_TYPED "${SRC_DIR}/py.typed")
# Stub-dir symlinks: _mpcf_cpu, _mpcf_cuda*, _mpcf (legacy)
file(GLOB _SRC_STUB_DIRS "${SRC_DIR}/_mpcf*")

foreach(_F IN LISTS _SRC_SO _SRC_TYPED)
  if(IS_SYMLINK "${_F}" OR EXISTS "${_F}")
    message(STATUS "Removing: ${_F}")
    file(REMOVE "${_F}")
  endif()
endforeach()

foreach(_D IN LISTS _SRC_STUB_DIRS)
  # Only remove directories/symlinks named _mpcf*, not _mpcf_cpp.py etc.
  if(IS_SYMLINK "${_D}")
    message(STATUS "Removing symlink: ${_D}")
    file(REMOVE "${_D}")
  elseif(IS_DIRECTORY "${_D}")
    message(STATUS "Removing directory: ${_D}")
    file(REMOVE_RECURSE "${_D}")
  endif()
endforeach()

# 2. Remove site-packages/masspcf contents
if(EXISTS "${SITE_DIR}")
  message(STATUS "Removing site-packages masspcf: ${SITE_DIR}")
  file(REMOVE_RECURSE "${SITE_DIR}")
endif()
