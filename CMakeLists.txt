cmake_minimum_required(VERSION 3.20)
project(avm-sycl-gpu-acceleration VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(AVM_ENABLE_SYCL "Enable SYCL GPU acceleration" ON)
option(AVM_BUILD_TESTS "Build test executables" ON)
option(AVM_BUILD_EXAMPLES "Build example programs" ON)

# Include SYCL configuration
include(cmake/sycl.cmake)

# Source files
set(SYCL_SOURCES
    src/sycl_context.cpp
    src/sycl_txfm.cpp
    src/sycl_me.cpp
    src/sycl_lpf.cpp
    src/sycl_intra.cpp
)

set(SYCL_HEADERS
    src/sycl_context.hpp
    src/sycl_txfm.hpp
    src/sycl_me.hpp
    src/sycl_lpf.hpp
    src/sycl_intra.hpp
    src/sycl_wrapper.hpp
)

# Main library
if(AVM_HAVE_SYCL)
    add_library(avm_sycl STATIC ${SYCL_SOURCES} ${SYCL_HEADERS})
    target_include_directories(avm_sycl PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(avm_sycl PRIVATE HAVE_SYCL=1)

    # SYCL specific flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
        target_compile_options(avm_sycl PRIVATE -fsycl)
    endif()

    message(STATUS "Building with SYCL GPU acceleration")
else()
    message(STATUS "SYCL not available - building stub library")
    add_library(avm_sycl INTERFACE)
endif()

# Tests
if(AVM_BUILD_TESTS AND AVM_HAVE_SYCL)
    enable_testing()

    add_executable(sycl_context_test tests/sycl_context_test.cpp)
    target_link_libraries(sycl_context_test PRIVATE avm_sycl)

    add_executable(sycl_txfm_test tests/sycl_txfm_test.cpp)
    target_link_libraries(sycl_txfm_test PRIVATE avm_sycl)

    add_executable(sycl_perf_test tests/sycl_perf_test.cpp)
    target_link_libraries(sycl_perf_test PRIVATE avm_sycl)

    # Add tests
    add_test(NAME SYCLContextTest COMMAND sycl_context_test)
    add_test(NAME SYCLTransformTest COMMAND sycl_txfm_test)
endif()

# Examples
if(AVM_BUILD_EXAMPLES AND AVM_HAVE_SYCL)
    add_executable(basic_example examples/basic_usage.cpp)
    target_link_libraries(basic_example PRIVATE avm_sycl)
endif()

# Install
include(GNUInstallDirs)

install(TARGETS avm_sycl
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${SYCL_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/avm/sycl
)

install(FILES cmake/sycl.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/avm_sycl
)
