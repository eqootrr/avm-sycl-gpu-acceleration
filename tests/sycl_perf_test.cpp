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
 * @file sycl_perf_test.cpp
 * @brief Performance benchmarks for SYCL-accelerated operations in AVM
 *
 * This file contains performance benchmarks comparing CPU vs SYCL implementations
 * of key video coding operations.
 */

#include <CL/sycl.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#if !defined(__SYCL_COMPILER_VERSION)
#error "SYCL compiler not detected. Please compile with -fsycl"
#endif

#include "config/av2_rtcd.h"
#include "test/acm_random.h"
#include "av2/common/av2_txfm.h"
#include "av2_ports/avm_timer.h"

using libavm_test::ACMRandom;

namespace libavm {
namespace sycl {
namespace test {

// Timing utilities
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

/**
 * @struct PerformanceResult
 * @brief Container for performance measurement results
 */
struct PerformanceResult {
  double cpu_time_ms;
  double sycl_time_ms;
  double speedup;
  size_t bytes_processed;
  double cpu_gbps;
  double sycl_gbps;

  void Print() const {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  CPU Time:     " << cpu_time_ms << " ms ("
              << cpu_gbps << " GB/s)" << std::endl;
    std::cout << "  SYCL Time:    " << sycl_time_ms << " ms ("
              << sycl_gbps << " GB/s)" << std::endl;
    std::cout << "  Speedup:      " << speedup << "x" << std::endl;
  }
};

/**
 * @class SYCLPerformanceTest
 * @brief Test fixture for SYCL performance benchmarks
 */
class SYCLPerformanceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    try {
      queue_ = std::make_unique<sycl::queue>(
          sycl::default_selector_v,
          {sycl::property::queue::enable_profiling{}});
    } catch (const sycl::exception& e) {
      GTEST_SKIP() << "SYCL device not available: " << e.what();
    }

    // Print device info
    auto device = queue_->get_device();
    std::cout << "\n=== Device Info ===" << std::endl;
    std::cout << "Name: "
              << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Max Compute Units: "
              << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "Global Memory: "
              << (device.get_info<sycl::info::device::global_mem_size>() /
                  1024 / 1024) << " MB" << std::endl;
    std::cout << "===================\n" << std::endl;
  }

  void TearDown() override {
    queue_.reset();
  }

  std::unique_ptr<sycl::queue> queue_;
};

/**
 * @test MemoryTransferPerformance
 * @brief Benchmark host-to-device and device-to-host memory transfers
 */
TEST_F(SYCLPerformanceTest, MemoryTransferPerformance) {
  ASSERT_NE(queue_, nullptr);

  std::vector<size_t> buffer_sizes = {
    1024,           // 1 KB
    1024 * 1024,    // 1 MB
    4 * 1024 * 1024, // 4 MB
    16 * 1024 * 1024 // 16 MB
  };

  constexpr int iterations = 100;

  std::cout << "\n=== Memory Transfer Performance ===" << std::endl;

  for (const auto buffer_size : buffer_sizes) {
    // Allocate host buffer
    std::vector<int> host_buffer(buffer_size);
    for (size_t i = 0; i < buffer_size; ++i) {
      host_buffer[i] = static_cast<int>(i);
    }

    // Allocate device buffer
    int* device_buffer = sycl::malloc_device<int>(buffer_size, *queue_);
    ASSERT_NE(device_buffer, nullptr);

    // Benchmark H2D transfer
    auto h2d_start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
      queue_->memcpy(device_buffer, host_buffer.data(),
                    buffer_size * sizeof(int));
      queue_->wait();
    }
    auto h2d_end = Clock::now();
    Duration h2d_time = h2d_end - h2d_start;

    // Benchmark D2H transfer
    std::vector<int> verify_buffer(buffer_size);
    auto d2h_start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
      queue_->memcpy(verify_buffer.data(), device_buffer,
                    buffer_size * sizeof(int));
      queue_->wait();
    }
    auto d2h_end = Clock::now();
    Duration d2h_time = d2h_end - d2h_start;

    // Calculate bandwidth
    double bytes_per_transfer = buffer_size * sizeof(int);
    double total_bytes = bytes_per_transfer * iterations;

    double h2d_bandwidth = (total_bytes / 1024 / 1024 / 1024) /
                          (h2d_time.count());
    double d2h_bandwidth = (total_bytes / 1024 / 1024 / 1024) /
                          (d2h_time.count());

    std::cout << "\nBuffer size: " << (buffer_size * sizeof(int) / 1024) << " KB" << std::endl;
    std::cout << "  H2D Bandwidth: " << h2d_bandwidth << " GB/s" << std::endl;
    std::cout << "  D2H Bandwidth: " << d2h_bandwidth << " GB/s" << std::endl;

    // Verify data integrity
    bool verified = true;
    for (size_t i = 0; i < buffer_size; ++i) {
      if (verify_buffer[i] != host_buffer[i]) {
        verified = false;
        break;
      }
    }
    EXPECT_TRUE(verified) << "Data verification failed for buffer size "
                          << buffer_size;

    sycl::free(device_buffer, *queue_);
  }

  std::cout << "\n====================================\n" << std::endl;
}

/**
 * @test KernelLaunchOverhead
 * @brief Measure kernel launch overhead
 */
TEST_F(SYCLPerformanceTest, KernelLaunchOverhead) {
  ASSERT_NE(queue_, nullptr);

  constexpr int iterations = 10000;
  constexpr int data_size = 1024;

  std::vector<int> data(data_size, 1);
  std::vector<int> result(data_size);

  sycl::buffer<int, 1> data_buf(data.data(), sycl::range<1>(data_size));
  sycl::buffer<int, 1> result_buf(result.data(), sycl::range<1>(data_size));

  // Warmup
  for (int i = 0; i < 100; ++i) {
    queue_->submit([&](sycl::handler& cgh) {
      auto in = data_buf.get_access<sycl::access::mode::read>(cgh);
      auto out = result_buf.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>(data_size),
          [=](sycl::id<1> idx) {
        out[idx] = in[idx] + 1;
          });
    });
  }
  queue_->wait();

  // Measure
  auto start = Clock::now();
  for (int i = 0; i < iterations; ++i) {
    queue_->submit([&](sycl::handler& cgh) {
      auto in = data_buf.get_access<sycl::access::mode::read>(cgh);
      auto out = result_buf.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>(data_size),
          [=](sycl::id<1> idx) {
        out[idx] = in[idx] + 1;
          });
    });
  }
  queue_->wait();
  auto end = Clock::now();

  Duration total_time = end - start;
  double avg_latency_us = (total_time.count() * 1e6) / iterations;

  std::cout << "\n=== Kernel Launch Overhead ===" << std::endl;
  std::cout << "Iterations: " << iterations << std::endl;
  std::cout << "Total time: " << total_time.count() * 1000 << " ms" << std::endl;
  std::cout << "Average latency: " << avg_latency_us << " us" << std::endl;
  std::cout << "=============================\n" << std::endl;

  // Verify results
  EXPECT_EQ(result[0], 1 + iterations);
}

/**
 * @test TransformPerformance
 * @brief Benchmark transform operation performance
 */
TEST_F(SYCLPerformanceTest, DISABLED_TransformPerformance) {
  ASSERT_NE(queue_, nullptr);

  std::vector<std::pair<TX_SIZE, int>> test_configs = {
    {TX_4X4, 4},
    {TX_8X8, 8},
    {TX_16X16, 16},
    {TX_32X32, 32}
  };

  constexpr int iterations = 1000;
  constexpr int bit_depth = 10;

  std::cout << "\n=== Transform Performance ===" << std::endl;

  for (const auto& [tx_size, size] : test_configs) {
    const int width = size;
    const int height = size;
    const int stride = size;
    const int num_coeffs = width * height;

    // Allocate buffers
    std::vector<int16_t> input(width * stride);
    std::vector<tran_low_t> output_ref(num_coeffs);
    std::vector<tran_low_t> output_sycl(num_coeffs);

    // Initialize with random data
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    for (int i = 0; i < width * stride; ++i) {
      input[i] = rnd.PseudoUniform(highbd_bit_depth);
    }

    // CPU performance
    avm_usec_timer cpu_timer;
    avm_usec_timer_start(&cpu_timer);
    for (int i = 0; i < iterations; ++i) {
      av2_fwd_txfm2d_c(input.data(), output_ref.data(), stride, DCT_DCT,
                       tx_size, bit_depth);
    }
    avm_usec_timer_mark(&cpu_timer);
    double cpu_time_ms = avm_usec_timer_elapsed(&cpu_timer) / 1000.0;

    // SYCL performance (placeholder - uses CPU for now)
    auto sycl_start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
      // TODO: Replace with actual SYCL kernel
      av2_fwd_txfm2d_c(input.data(), output_sycl.data(), stride, DCT_DCT,
                       tx_size, bit_depth);
    }
    auto sycl_end = Clock::now();
    double sycl_time_ms = Duration(sycl_end - sycl_start).count() * 1000;

    // Calculate metrics
    size_t bytes_per_transform = (width * stride + num_coeffs) * sizeof(int16_t);
    size_t total_bytes = bytes_per_transform * iterations;

    PerformanceResult result;
    result.cpu_time_ms = cpu_time_ms;
    result.sycl_time_ms = sycl_time_ms;
    result.speedup = cpu_time_ms / sycl_time_ms;
    result.bytes_processed = total_bytes;
    result.cpu_gbps = (total_bytes / 1024.0 / 1024 / 1024) / (cpu_time_ms / 1000);
    result.sycl_gbps = (total_bytes / 1024.0 / 1024 / 1024) / (sycl_time_ms / 1000);

    std::cout << "\nTransform size: " << size << "x" << size << std::endl;
    result.Print();
  }

  std::cout << "\n===============================\n" << std::endl;
}

/**
 * @test USMVsBufferPerformance
 * @brief Compare USM vs Buffer performance
 */
TEST_F(SYCLPerformanceTest, USMVsBufferPerformance) {
  ASSERT_NE(queue_, nullptr);

  constexpr int data_size = 1024 * 1024;  // 1M elements
  constexpr int iterations = 100;

  std::cout << "\n=== USM vs Buffer Performance ===" << std::endl;

  // Initialize data
  std::vector<int> host_data(data_size);
  for (int i = 0; i < data_size; ++i) {
    host_data[i] = i;
  }

  // Test USM performance
  int* usm_input = sycl::malloc_shared<int>(data_size, *queue_);
  int* usm_output = sycl::malloc_shared<int>(data_size, *queue_);

  std::copy(host_data.begin(), host_data.end(), usm_input);

  auto usm_start = Clock::now();
  for (int i = 0; i < iterations; ++i) {
    queue_->submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<1>(data_size),
          [=](sycl::id<1> idx) {
        usm_output[idx] = usm_input[idx] * 2;
          });
    });
  }
  queue_->wait();
  auto usm_end = Clock::now();
  double usm_time_ms = Duration(usm_end - usm_start).count() * 1000;

  // Test Buffer performance
  sycl::buffer<int, 1> buf_input(host_data.data(), sycl::range<1>(data_size));
  std::vector<int> buf_output(data_size);
  sycl::buffer<int, 1> buf_output_buf(buf_output.data(), sycl::range<1>(data_size));

  auto buf_start = Clock::now();
  for (int i = 0; i < iterations; ++i) {
    queue_->submit([&](sycl::handler& cgh) {
      auto in = buf_input.get_access<sycl::access::mode::read>(cgh);
      auto out = buf_output_buf.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>(data_size),
          [=](sycl::id<1> idx) {
        out[idx] = in[idx] * 2;
          });
    });
  }
  queue_->wait();
  auto buf_end = Clock::now();
  double buf_time_ms = Duration(buf_end - buf_start).count() * 1000;

  std::cout << "Data size: " << (data_size * sizeof(int) / 1024) << " KB" << std::endl;
  std::cout << "Iterations: " << iterations << std::endl;
  std::cout << "USM time:  " << usm_time_ms << " ms" << std::endl;
  std::cout << "Buffer time: " << buf_time_ms << " ms" << std::endl;
  std::cout << "Ratio (USM/Buffer): " << (usm_time_ms / buf_time_ms) << "x" << std::endl;

  // Verify both produce same results
  bool verified = true;
  for (int i = 0; i < data_size; i += data_size / 100) {  // Sample check
    if (usm_output[i] != buf_output[i]) {
      verified = false;
      break;
    }
  }
  EXPECT_TRUE(verified) << "USM and Buffer results differ";

  sycl::free(usm_input, *queue_);
  sycl::free(usm_output, *queue_);

  std::cout << "\n==================================\n" << std::endl;
}

/**
 * @test ScalingPerformance
 * @brief Test performance scaling with work group size
 */
TEST_F(SYCLPerformanceTest, ScalingPerformance) {
  ASSERT_NE(queue_, nullptr);

  constexpr int data_size = 1024 * 1024;
  std::vector<int> input(data_size, 1);
  std::vector<int> output(data_size);

  sycl::buffer<int, 1> input_buf(input.data(), sycl::range<1>(data_size));
  sycl::buffer<int, 1> output_buf(output.data(), sycl::range<1>(data_size));

  std::vector<size_t> group_sizes = {32, 64, 128, 256, 512};

  std::cout << "\n=== Work Group Size Scaling ===" << std::endl;

  for (const auto group_size : group_sizes) {
    const size_t num_groups = (data_size + group_size - 1) / group_size;

    auto start = Clock::now();
    queue_->submit([&](sycl::handler& cgh) {
      auto in = input_buf.get_access<sycl::access::mode::read>(cgh);
      auto out = output_buf.get_access<sycl::access::mode::write>(cgh);
      sycl::nd_range<1> range(data_size, group_size);
      cgh.parallel_for(range,
          [=](sycl::nd_item<1> item) {
        out[item.get_global_id()] = in[item.get_global_id()] * 2;
          });
    });
    queue_->wait();

    auto end = Clock::now();
    double time_ms = Duration(end - start).count() * 1000;

    std::cout << "Group size: " << std::setw(4) << group_size
              << "  Time: " << std::setw(8) << time_ms << " ms"
              << "  Throughput: " << std::setw(6) << std::fixed
              << std::setprecision(2)
              << (data_size * sizeof(int) / 1024.0 / 1024) / (time_ms / 1000)
              << " GB/s" << std::endl;
  }

  std::cout << "\n==================================\n" << std::endl;
}

/**
 * @test EventProfiling
 * @brief Demonstrate event-based profiling
 */
TEST_F(SYCLPerformanceTest, EventProfiling) {
  ASSERT_NE(queue_, nullptr);

  constexpr int data_size = 1024 * 1024;
  std::vector<int> input(data_size, 1);
  std::vector<int> output(data_size);

  sycl::buffer<int, 1> input_buf(input.data(), sycl::range<1>(data_size));
  sycl::buffer<int, 1> output_buf(output.data(), sycl::range<1>(data_size));

  std::cout << "\n=== Event Profiling ===" << std::endl;

  auto event = queue_->submit([&](sycl::handler& cgh) {
    auto in = input_buf.get_access<sycl::access::mode::read>(cgh);
    auto out = output_buf.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for(sycl::range<1>(data_size),
        [=](sycl::id<1> idx) {
      out[idx] = in[idx] * 2;
        });
  });

  event.wait();

  auto submit_time =
      event.get_profiling_info<sycl::info::event_profiling::command_submit>();
  auto start_time =
      event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end_time =
      event.get_profiling_info<sycl::info::event_profiling::command_end>();

  double submit_ns = (start_time - submit_time) / 1000.0;
  double exec_ns = (end_time - start_time) / 1000.0;

  std::cout << "Submit to start latency: " << std::fixed << std::setprecision(2)
            << submit_ns << " us" << std::endl;
  std::cout << "Execution time: " << exec_ns << " us" << std::endl;
  std::cout << "Throughput: "
            << (data_size * sizeof(int) / 1024.0 / 1024) / (exec_ns / 1e6)
            << " GB/s" << std::endl;

  std::cout << "\n=======================\n" << std::endl;
}

}  // namespace test
}  // namespace sycl
}  // namespace libavm

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
