# avm_dsp/sycl/CMakeLists.txt
# SYCL GPU acceleration module

if(NOT HAVE_SYCL)
    return()
endif()

set(AVM_SYCL_SOURCES
    sycl_context.cpp
    sycl_txfm.cpp
    sycl_me.cpp
    sycl_lpf.cpp
    sycl_intra.cpp
)

# Create SYCL library
add_library(avm_sycl STATIC ${AVM_SYCL_SOURCES})

target_include_directories(avm_sycl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AVM_ROOT}
)

# SYCL specific flags
if(SYCL_BACKEND STREQUAL "METAL")
    target_compile_options(avm_sycl PRIVATE -fsycl -fsycl-targets=spir64_gen)
elseif(SYCL_BACKEND STREQUAL "CUDA")
    target_compile_options(avm_sycl PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
else()
    target_compile_options(avm_sycl PRIVATE -fsycl -fsycl-targets=spir64_gen)
endif()

target_link_libraries(avm_sycl PUBLIC avm)
