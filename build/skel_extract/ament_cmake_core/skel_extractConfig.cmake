# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_skel_extract_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED skel_extract_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(skel_extract_FOUND FALSE)
  elseif(NOT skel_extract_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(skel_extract_FOUND FALSE)
  endif()
  return()
endif()
set(_skel_extract_CONFIG_INCLUDED TRUE)

# output package information
if(NOT skel_extract_FIND_QUIETLY)
  message(STATUS "Found skel_extract: 0.0.0 (${skel_extract_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'skel_extract' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${skel_extract_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(skel_extract_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${skel_extract_DIR}/${_extra}")
endforeach()
