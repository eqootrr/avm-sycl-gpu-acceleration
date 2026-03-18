# Copyright (c) 2021, Alliance for Open Media. All rights reserved
#
# This source code is subject to the terms of the BSD 3-Clause Clear License
# and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
# License was not distributed with this source code in the LICENSE file, you
# can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
# Alliance for Open Media Patent License 1.0 was not distributed with this
# source code in the PATENTS file, you can obtain it at
# aomedia.org/license/patent-license/.

cmake_minimum_required(VERSION 3.20)

# Find SYCL compiler
find_package(IntelSYCL CONFIG QUIET)

if(TARGET sycl)
  set(AVM_HAVE_SYCL 1 CACHE INTERNAL "SYCL support available")
  message(STATUS "SYCL support: enabled")
else()
  set(AVM_HAVE_SYCL 0 CACHE INTERNAL "SYCL support available")
  message(STATUS "SYCL support: disabled - IntelSYCL not found")
  return()
endif()

# SYCL test sources
set(AVM_SYCL_TEST_SOURCES
  sycl/sycl_context_test.cpp
  sycl/sycl_txfm_test.cpp
  sycl/sycl_perf_test.cpp
)

# Create SYCL test executables
foreach(test_source ${AVM_SYCL_TEST_SOURCES})
  get_filename_component(test_name ${test_source} NAME_WE)

  add_executable(${test_name} ${test_source})

  # Set compiler flags for SYCL
  target_compile_options(${test_name} PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,IntelLLVM>:-fsycl>
  )

  # Link against AVM libraries
  target_link_libraries(${test_name} PRIVATE
    avm_common
    avm_encoder
    avm_decoder
    gtest
    gtest_main
  )

  # Add test to CTest
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# SYCL test targets
add_custom_target(sycl_tests)
foreach(test_source ${AVM_SYCL_TEST_SOURCES})
  get_filename_component(test_name ${test_source} NAME_WE)
  add_dependencies(sycl_tests ${test_name})
endforeach()

# Print configuration summary
message(STATUS "SYCL Test Configuration:")
message(STATUS "  Test sources: ${AVM_SYCL_TEST_SOURCES}")
message(STATUS "  IntelSYCL found: ${TARGET sycl}")
