# Copyright (c) 2021, Alliance for Open Media. All rights reserved
#
# This source code is subject to the terms of the BSD 3-Clause Clear License and
# the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
# License was not distributed with this source code in the LICENSE file, you
# can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
# Alliance for Open Media Patent License 1.0 was not distributed with this
# source code in the PATENTS file, you can obtain it at
# aomedia.org/license/patent-license/.

#
# SYCL support configuration for AVM
#

# Option to enable SYCL support
option(AVM_ENABLE_SYCL "Enable SYCL acceleration" OFF)

# Check for SYCL compiler support
if(AVM_ENABLE_SYCL)
  # Try to find Intel DPC++ compiler or other SYCL implementations
  find_package(IntelSYCL CONFIG QUIET)

  if(TARGET sycl OR CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Clang")
    # Check if compiler supports SYCL
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-fsycl")
    check_cxx_source_compiles("
      #include <CL/sycl.hpp>
      int main() {
        sycl::queue q;
        return 0;
      }
    " AVM_SYCL_COMPILER_CHECK)

    if(AVM_SYCL_COMPILER_CHECK OR TARGET sycl)
      set(AVM_HAVE_SYCL 1 CACHE INTERNAL "SYCL support available")
      message(STATUS "SYCL support: enabled")

      # Set SYCL compile flags
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsycl")

      # SYCL target device
      if(NOT AVM_SYCL_TARGET_DEVICE)
        set(AVM_SYCL_TARGET_DEVICE "spir64" CACHE STRING "SYCL target device")
      endif()

      message(STATUS "SYCL target device: ${AVM_SYCL_TARGET_DEVICE}")
    else()
      message(WARNING "SYCL compiler check failed. Disabling SYCL support.")
      set(AVM_HAVE_SYCL 0 CACHE INTERNAL "SYCL support available")
    endif()
  else()
    message(STATUS "SYCL support: disabled - IntelSYCL not found and compiler does not support -fsycl")
    set(AVM_HAVE_SYCL 0 CACHE INTERNAL "SYCL support available")
  endif()
else()
  message(STATUS "SYCL support: disabled by user")
  set(AVM_HAVE_SYCL 0 CACHE INTERNAL "SYCL support available")
endif()

# Create config flag for SYCL
if(AVM_HAVE_SYCL)
  set(AVM_HAVE_SYCL_TXFM 1 CACHE INTERNAL "SYCL transform support")
else()
  set(AVM_HAVE_SYCL_TXFM 0 CACHE INTERNAL "SYCL transform support")
endif()

# Helper function to add SYCL sources to a target
function(avm_add_sycl_sources target)
  if(AVM_HAVE_SYCL)
    target_sources(${target} PRIVATE ${ARGN})
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANG_AND_ID:CXX,IntelLLVM>:-fsycl>
      $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fsycl>
    )
  endif()
endfunction()

# Helper function to link SYCL libraries
function(avm_link_sycl target)
  if(AVM_HAVE_SYCL)
    if(TARGET sycl)
      target_link_libraries(${target} PRIVATE sycl)
    endif()
  endif()
endfunction()
