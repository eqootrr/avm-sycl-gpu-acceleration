/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

/**
 * @file sycl_context_test.cpp
 * @brief Unit tests for SYCL context management in AVM
 *
 * This file tests the SYCL context creation, device selection, and
 * resource management for accelerated video coding operations.
 */

#include <CL/sycl.hpp>
#include <memory>
#include <vector>

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#if !defined(__SYCL_COMPILER_VERSION)
#error "SYCL compiler not detected. Please compile with -fsycl"
#endif

namespace libavm {
namespace sycl {
namespace test {

/**
 * @class SYCLContextTest
 * @brief Test fixture for SYCL context management tests
 */
class SYCLContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize default SYCL queue
    try {
      queue_ = std::make_unique<sycl::queue>(sycl::default_selector_v);
      ASSERT_NE(queue_, nullptr) << "Failed to create SYCL queue";
    } catch (const sycl::exception& e) {
      GTEST_SKIP() << "SYCL device not available: " << e.what();
    }
  }

  void TearDown() override {
    queue_.reset();
  }

  std::unique_ptr<sycl::queue> queue_;
};

/**
 * @test QueueCreation
 * @brief Test that SYCL queue can be created with default selector
 */
TEST_F(SYCLContextTest, QueueCreation) {
  ASSERT_NE(queue_, nullptr);
  EXPECT_NO_THROW({
    auto device = queue_->get_device();
    auto platform = device.get_platform();

    std::cout << "SYCL Device: "
              << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "SYCL Platform: "
              << platform.get_info<sycl::info::platform::name>() << std::endl;
  });
}

/**
 * @test DeviceInfo
 * @brief Test retrieval of device capabilities
 */
TEST_F(SYCLContextTest, DeviceInfo) {
  ASSERT_NE(queue_, nullptr);

  auto device = queue_->get_device();

  // Check device is available
  EXPECT_TRUE(device.is_gpu() || device.is_cpu() || device.is_accelerator());

  // Get device info
  const auto max_compute_units =
      device.get_info<sycl::info::device::max_compute_units>();
  const auto global_mem_size =
      device.get_info<sycl::info::device::global_mem_size>();
  const auto local_mem_size =
      device.get_info<sycl::info::device::local_mem_size>();

  EXPECT_GT(max_compute_units, 0);
  EXPECT_GT(global_mem_size, 0);
  EXPECT_GT(local_mem_size, 0);

  std::cout << "Max Compute Units: " << max_compute_units << std::endl;
  std::cout << "Global Memory: " << (global_mem_size / 1024 / 1024) << " MB" << std::endl;
  std::cout << "Local Memory: " << (local_mem_size / 1024) << " KB" << std::endl;
}

/**
 * @test USMAllocation
 * @brief Test Unified Shared Memory allocation
 */
TEST_F(SYCLContextTest, USMAllocation) {
  ASSERT_NE(queue_, nullptr);

  constexpr size_t buffer_size = 1024;

  // Test USM allocation
  int* ptr = nullptr;
  EXPECT_NO_THROW({
    ptr = sycl::malloc_shared<int>(buffer_size, *queue_);
  });

  ASSERT_NE(ptr, nullptr);

  // Write to buffer
  for (size_t i = 0; i < buffer_size; ++i) {
    ptr[i] = static_cast<int>(i);
  }

  // Verify on host
  for (size_t i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(ptr[i], static_cast<int>(i));
  }

  // Free memory
  EXPECT_NO_THROW({
    sycl::free(ptr, *queue_);
  });
}

/**
 * @test BufferAllocation
 * @brief Test SYCL buffer allocation and access
 */
TEST_F(SYCLContextTest, BufferAllocation) {
  ASSERT_NE(queue_, nullptr);

  constexpr size_t buffer_size = 1024;
  std::vector<int> host_data(buffer_size);

  // Initialize host data
  for (size_t i = 0; i < buffer_size; ++i) {
    host_data[i] = static_cast<int>(i * 2);
  }

  // Test buffer operations
  EXPECT_NO_THROW({
    sycl::buffer<int, 1> buf(host_data.data(), sycl::range<1>(buffer_size));

    queue_->submit([&](sycl::handler& cgh) {
      auto accessor = buf.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::range<1>(buffer_size),
          [=](sycl::id<1> idx) {
        accessor[idx] *= 2;
      });
    });

    queue_->wait_and_throw();
  });

  // Verify results
  for (size_t i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(host_data[i], static_cast<int>(i * 4));
  }
}

/**
 * @test KernelExecution
 * @brief Test basic kernel execution
 */
TEST_F(SYCLContextTest, KernelExecution) {
  ASSERT_NE(queue_, nullptr);

  constexpr size_t size = 256;
  std::vector<int> input(size, 1);
  std::vector<int> output(size, 0);

  {
    sycl::buffer<int, 1> input_buf(input.data(), sycl::range<1>(size));
    sycl::buffer<int, 1> output_buf(output.data(), sycl::range<1>(size));

    queue_->submit([&](sycl::handler& cgh) {
      auto in = input_buf.get_access<sycl::access::mode::read>(cgh);
      auto out = output_buf.get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for(sycl::range<1>(size),
          [=](sycl::id<1> idx) {
        out[idx] = in[idx] * 2 + 1;
      });
    });

    queue_->wait_and_throw();
  }

  // Verify kernel executed correctly
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(output[i], 3);  // 1 * 2 + 1 = 3
  }
}

/**
 * @test LocalMemory
 * @brief test local memory usage in kernels
 */
TEST_F(SYCLContextTest, LocalMemory) {
  ASSERT_NE(queue_, nullptr);

  auto device = queue_->get_device();
  const auto local_mem_size =
      device.get_info<sycl::info::device::local_mem_size>();

  if (local_mem_size == 0) {
    GTEST_SKIP() << "Device does not support local memory";
  }

  constexpr size_t size = 256;
  constexpr size_t group_size = 64;
  std::vector<int> data(size, 1);
  std::vector<int> results(size);

  {
    sycl::buffer<int, 1> data_buf(data.data(), sycl::range<1>(size));
    sycl::buffer<int, 1> result_buf(results.data(), sycl::range<1>(size));

    sycl::range<1> global_size(size);
    sycl::range<1> local_range(group_size);

    queue_->submit([&](sycl::handler& cgh) {
      auto in = data_buf.get_access<sycl::access::mode::read>(cgh);
      auto out = result_buf.get_access<sycl::access::mode::write>(cgh);

      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::target::local> local_mem(sycl::range<1>(group_size), cgh);

      cgh.parallel_for(sycl::nd_range<1>(global_size, local_range),
          [=](sycl::nd_item<1> item) {
            auto lid = item.get_local_id();
            auto gid = item.get_global_id();

            // Load to local memory
            local_mem[lid] = in[gid];

            item.barrier(sycl::access::fence_space::local_space);

            // Store back
            out[gid] = local_mem[(lid + 1) % group_size];
          });
    });

    queue_->wait_and_throw();
  }

  // Verify local memory shuffle worked
  bool all_valid = true;
  for (size_t i = 0; i < size; ++i) {
    if (results[i] != 1) all_valid = false;
  }
  EXPECT_TRUE(all_valid);
}

/**
 * @test MultipleQueues
 * @brief Test multiple queue creation and synchronization
 */
TEST_F(SYCLContextTest, MultipleQueues) {
  ASSERT_NE(queue_, nullptr);

  // Create additional queues
  std::unique_ptr<sycl::queue> queue2, queue3;

  EXPECT_NO_THROW({
    queue2 = std::make_unique<sycl::queue>(sycl::default_selector_v);
    queue3 = std::make_unique<sycl::queue>(sycl::default_selector_v);
  });

  ASSERT_NE(queue2, nullptr);
  ASSERT_NE(queue3, nullptr);

  // Execute work on all queues
  constexpr size_t size = 128;
  std::vector<int> data1(size, 1);
  std::vector<int> data2(size, 2);
  std::vector<int> data3(size, 3);

  auto process = [&](sycl::queue& q, std::vector<int>& data, int multiplier) {
    sycl::buffer<int, 1> buf(data.data(), sycl::range<1>(size));
    q.submit([&](sycl::handler& cgh) {
      auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.parallel_for(sycl::range<1>(size),
          [=](sycl::id<1> idx) {
        acc[idx] *= multiplier;
      });
    });
  };

  process(*queue_, data1, 2);
  process(*queue2, data2, 3);
  process(*queue3, data3, 4);

  // Wait for all queues
  queue_->wait_and_throw();
  queue2->wait_and_throw();
  queue3->wait_and_throw();

  EXPECT_EQ(data1[0], 2);
  EXPECT_EQ(data2[0], 6);
  EXPECT_EQ(data3[0], 12);
}

/**
 * @test AspectQuery
 * @brief Test device aspect/feature query
 */
TEST_F(SYCLContextTest, AspectQuery) {
  ASSERT_NE(queue_, nullptr);

  auto device = queue_->get_device();

  // Query common aspects
  std::vector<sycl::aspect> aspects = {
    sycl::aspect::cpu,
    sycl::aspect::gpu,
    sycl::aspect::accelerator,
    sycl::aspect::custom,
    sycl::aspect::fp16,
    sycl::aspect::fp64,
    sycl::aspect::atomic64,
    sycl::aspect::image,
    sycl::aspect::online_compiler,
    sycl::aspect::online_linker,
    sycl::aspect::queue_profiling,
    sycl::aspect::usm_device_allocations,
    sycl::aspect::usm_host_allocations,
    sycl::aspect::usm_shared_allocations,
    sycl::aspect::usm_system_allocator
  };

  for (const auto& asp : aspects) {
    bool has_aspect = device.has(asp);
    if (has_aspect) {
      std::cout << "Device supports: "
                << static_cast<int>(asp) << std::endl;
    }
  }

  // Verify USM support
  EXPECT_TRUE(device.has(sycl::aspect::usm_shared_allocations) ||
              device.has(sycl::aspect::usm_device_allocations));
}

}  // namespace test
}  // namespace sycl
}  // namespace libavm

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
