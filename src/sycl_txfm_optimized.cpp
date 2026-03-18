/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved
 *
 * Optimized SYCL transform kernels for AVM encoder
 *
 * Performance optimizations:
 * 1. Single kernel launch for 2D transforms (row+column in one kernel)
 * 2. Local memory tiling to minimize global memory access
 * 3. Proper work-group utilization (all work-items contribute)
 * 4. Vectorized memory access when possible
 * 5. Batch processing to amortize kernel launch overhead
 */

#ifdef HAVE_SYCL

#include "sycl_txfm_optimized.hpp"
#include "sycl_context.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>

namespace avm {
namespace sycl {

namespace {

// ============================================================================
// Fixed-point DCT coefficients (scaled by 2^14)
// ============================================================================

constexpr int kCosPi16_16384 = 16069;   // cos(pi/16) * 16384
constexpr int kCos2Pi16_16384 = 15137;  // cos(2*pi/16) * 16384
constexpr int kCos3Pi16_16384 = 13623;  // cos(3*pi/16) * 16384
constexpr int kCos4Pi16_16384 = 11585;  // cos(4*pi/16) * 16384
constexpr int kCos5Pi16_16384 = 9102;   // cos(5*pi/16) * 16384
constexpr int kCos6Pi16_16384 = 6270;   // cos(6*pi/16) * 16384
constexpr int kCos7Pi16_16384 = 3196;   // cos(7*pi/16) * 16384

constexpr int kCosPi8_16384 = 15137;    // cos(pi/8) * 16384
constexpr int kSinPi8_16384 = 8867;     // sin(pi/8) * 16384
constexpr int kCos3Pi8_16384 = 6270;    // cos(3*pi/8) * 16384
constexpr int kSin3Pi8_16384 = 13623;   // sin(3*pi/8) * 16384

// Shift amounts for 8x8 forward transform
constexpr int kFwdShift8x8_0 = 2;
constexpr int kFwdShift8x8_1 = 1;
constexpr int kFwdShift8x8_2 = 1;

// Shift amounts for 8x8 inverse transform
constexpr int kInvShift8x8_0 = 1;
constexpr int kInvShift8x8_1 = 2;
constexpr int kInvShift8x8_2 = 1;

// ============================================================================
// Device-side kernel functions
// ============================================================================

// Forward 8-point DCT-II (Chen's algorithm)
// Operates on local memory array
struct FDCT8Kernel {
    // 1D forward DCT on 8-point array
    static void compute(int* data) {
        // Stage 1: Even/odd separation
        int s0 = data[0] + data[7];
        int s1 = data[1] + data[6];
        int s2 = data[2] + data[5];
        int s3 = data[3] + data[4];
        int s4 = data[0] - data[7];
        int s5 = data[1] - data[6];
        int s6 = data[2] - data[5];
        int s7 = data[3] - data[4];

        // Even part
        int even0 = s0 + s3;
        int even1 = s1 + s2;
        int even2 = s1 - s2;
        int even3 = s0 - s3;

        int step[8];
        step[0] = even0 + even1;
        step[4] = even0 - even1;
        step[2] = (kCos2Pi16_16384 * even2 + kCos6Pi16_16384 * even3) >> 14;
        step[6] = (kCos6Pi16_16384 * even2 - kCos2Pi16_16384 * even3) >> 14;

        // Odd part
        step[1] = (kCosPi16_16384 * s4 + kCos7Pi16_16384 * s7) >> 14;
        step[7] = (kCos7Pi16_16384 * s4 - kCosPi16_16384 * s7) >> 14;
        step[3] = (kCos3Pi16_16384 * s5 + kCos5Pi16_16384 * s6) >> 14;
        step[5] = (kCos5Pi16_16384 * s5 - kCos3Pi16_16384 * s6) >> 14;

        // Final butterfly
        int odd0 = step[1] + step[3];
        int odd1 = step[1] - step[3];
        int odd2 = step[5] - step[7];
        int odd3 = step[5] + step[7];

        data[0] = step[0];
        data[1] = odd0;
        data[2] = step[2];
        data[3] = odd2;
        data[4] = step[4];
        data[5] = odd3;
        data[6] = step[6];
        data[7] = odd1;
    }
};

// Inverse 8-point DCT-III
struct IDCT8Kernel {
    static void compute(int* data) {
        int step[8];

        // Reconstruct odd part
        int odd0 = data[1];
        int odd1 = data[7];
        int odd2 = data[3];
        int odd3 = data[5];

        step[1] = odd0 + odd1;
        step[3] = odd0 - odd1;
        step[5] = odd3 - odd2;
        step[7] = odd3 + odd2;

        // Inverse rotation
        int s4 = (kCosPi16_16384 * step[1] - kCos7Pi16_16384 * step[7]) >> 14;
        int s7 = (kCos7Pi16_16384 * step[1] + kCosPi16_16384 * step[7]) >> 14;
        int s5 = (kCos3Pi16_16384 * step[3] - kCos5Pi16_16384 * step[5]) >> 14;
        int s6 = (kCos5Pi16_16384 * step[3] + kCos3Pi16_16384 * step[5]) >> 14;

        // Even part
        int even0 = data[0];
        int even1 = data[4];
        int even2 = data[2];
        int even3 = data[6];

        int s0 = even0 + even1;
        int s3 = even0 - even1;

        int s1 = (kCos2Pi16_16384 * even2 - kCos6Pi16_16384 * even3) >> 14;
        int s2 = (kCos6Pi16_16384 * even2 + kCos2Pi16_16384 * even3) >> 14;

        // Final combination
        data[0] = s0 + s1;
        data[1] = s4 + s5;
        data[2] = s2 + s3;
        data[3] = s6 + s7;
        data[4] = s3 - s2;
        data[5] = s7 - s6;
        data[6] = s1 - s0;
        data[7] = s5 - s4;
    }
};

// Forward 4-point DCT-II
struct FDCT4Kernel {
    static void compute(int* data) {
        int s0 = data[0] + data[3];
        int s1 = data[1] + data[2];
        int s2 = data[0] - data[3];
        int s3 = data[1] - data[2];

        int even0 = s0 + s1;
        int even1 = s0 - s1;

        int odd0 = (kCosPi8_16384 * s2 + kSinPi8_16384 * s3) >> 14;
        int odd1 = (kSinPi8_16384 * s2 - kCosPi8_16384 * s3) >> 14;

        data[0] = even0;
        data[1] = odd0;
        data[2] = even1;
        data[3] = odd1;
    }
};

// Inverse 4-point DCT-III
struct IDCT4Kernel {
    static void compute(int* data) {
        int even0 = data[0] + data[2];
        int even1 = data[0] - data[2];

        int odd0 = data[1];
        int odd1 = data[3];

        int s2 = (kCosPi8_16384 * odd0 - kSinPi8_16384 * odd1) >> 14;
        int s3 = (kSinPi8_16384 * odd0 + kCosPi8_16384 * odd1) >> 14;

        data[0] = even0 + s2;
        data[1] = even1 + s3;
        data[2] = even1 - s3;
        data[3] = even0 - s2;
    }
};

// Round and shift helper
inline int round_shift(int value, int shift) {
    if (shift <= 0) return value;
    return (value + (1 << (shift - 1))) >> shift;
}

// Clamp to pixel range
inline uint16_t clamp_pixel(int value, int bd) {
    const int max_val = (1 << bd) - 1;
    return static_cast<uint16_t>(sycl::clamp(value, 0, max_val));
}

// ============================================================================
// Performance profiling state
// ============================================================================

bool g_profiling_enabled = false;
TxfmPerfStats g_last_stats{};

}  // anonymous namespace

// ============================================================================
// Optimized 8x8 Forward DCT Implementation
// ============================================================================

void fdct8x8_optimized(::sycl::queue& q,
                       const int16_t* input,
                       tran_low_t* output,
                       int stride,
                       const TxfmParams& params) {
    constexpr int kSize = 8;
    constexpr int kNumThreads = kSize * kSize;

    auto start = std::chrono::high_resolution_clock::now();

    // Single kernel launch for both row and column transforms
    // Uses local memory tile for efficient 2D transform
    q.submit([&](::sycl::handler& cgh) {
        // Local memory tile for the 8x8 block
        ::sycl::accessor<int, 2, ::sycl::access::mode::read_write,
                         ::sycl::target::local> local_tile(::sycl::range<2>(kSize, kSize), cgh);

        cgh.parallel_for(::sycl::nd_range<2>(::sycl::range<2>(kSize, kSize),
                                              ::sycl::range<2>(kSize, kSize)),
            [=](::sycl::nd_item<2> item) {
                const int row = item.get_global_id(0);
                const int col = item.get_global_id(1);
                const int local_row = item.get_local_id(0);
                const int local_col = item.get_local_id(1);

                // Load input to local memory with coalesced access
                // Each work-item loads one element
                local_tile[local_row][local_col] =
                    static_cast<int>(input[row * stride + col]) << kFwdShift8x8_0;

                // Barrier to ensure all data is loaded
                item.barrier(::sycl::access::fence_space::local_space);

                // === Row transform ===
                // Each row is processed by its corresponding work-items
                int row_data[8];

                // Load row from local tile
                for (int c = 0; c < kSize; ++c) {
                    row_data[c] = local_tile[local_row][c];
                }

                // Apply 1D DCT on row
                FDCT8Kernel::compute(row_data);

                // Store row back to local tile with shift
                for (int c = 0; c < kSize; ++c) {
                    local_tile[local_row][c] = round_shift(row_data[c], kFwdShift8x8_1);
                }

                // Barrier after row transform
                item.barrier(::sycl::access::fence_space::local_space);

                // === Column transform ===
                // Each column is processed by its corresponding work-items
                int col_data[8];

                // Load column from local tile
                for (int r = 0; r < kSize; ++r) {
                    col_data[r] = local_tile[r][local_col];
                }

                // Apply 1D DCT on column
                FDCT8Kernel::compute(col_data);

                // Store to global output with final shift
                output[row * kSize + col] =
                    static_cast<tran_low_t>(round_shift(col_data[local_row], kFwdShift8x8_2));
            });
    }).wait();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if (g_profiling_enabled) {
        g_last_stats.total_time_ms = duration.count() / 1000.0;
        g_last_stats.avg_time_per_block_us = duration.count();
        g_last_stats.throughput_blocks_per_sec = 1000000.0 / duration.count();
        g_last_stats.memory_bandwidth_gbps =
            (2.0 * kSize * kSize * sizeof(int16_t)) / (duration.count() / 1e9) / 1e9;
    }
}

// ============================================================================
// Optimized 4x4 Forward DCT Implementation
// ============================================================================

void fdct4x4_optimized(::sycl::queue& q,
                       const int16_t* input,
                       tran_low_t* output,
                       int stride,
                       const TxfmParams& params) {
    constexpr int kSize = 4;

    q.submit([&](::sycl::handler& cgh) {
        ::sycl::accessor<int, 2, ::sycl::access::mode::read_write,
                         ::sycl::target::local> local_tile(::sycl::range<2>(kSize, kSize), cgh);

        cgh.parallel_for(::sycl::nd_range<2>(::sycl::range<2>(kSize, kSize),
                                              ::sycl::range<2>(kSize, kSize)),
            [=](::sycl::nd_item<2> item) {
                const int row = item.get_global_id(0);
                const int col = item.get_global_id(1);
                const int local_row = item.get_local_id(0);
                const int local_col = item.get_local_id(1);

                // Load input to local memory
                local_tile[local_row][local_col] =
                    static_cast<int>(input[row * stride + col]);

                item.barrier(::sycl::access::fence_space::local_space);

                // Row transform
                int row_data[4];
                for (int c = 0; c < kSize; ++c) {
                    row_data[c] = local_tile[local_row][c];
                }
                FDCT4Kernel::compute(row_data);
                for (int c = 0; c < kSize; ++c) {
                    local_tile[local_row][c] = row_data[c];
                }

                item.barrier(::sycl::access::fence_space::local_space);

                // Column transform
                int col_data[4];
                for (int r = 0; r < kSize; ++r) {
                    col_data[r] = local_tile[r][local_col];
                }
                FDCT4Kernel::compute(col_data);

                // Store to global output with shift
                output[row * kSize + col] =
                    static_cast<tran_low_t>(round_shift(col_data[local_row], 1));
            });
    }).wait();
}

// ============================================================================
// Optimized 8x8 Inverse DCT Implementation
// ============================================================================

void idct8x8_optimized(::sycl::queue& q,
                       const tran_low_t* input,
                       uint16_t* output,
                       int stride,
                       const TxfmParams& params) {
    constexpr int kSize = 8;
    const int bd = params.bd;

    q.submit([&](::sycl::handler& cgh) {
        ::sycl::accessor<int, 2, ::sycl::access::mode::read_write,
                         ::sycl::target::local> local_tile(::sycl::range<2>(kSize, kSize), cgh);

        cgh.parallel_for(::sycl::nd_range<2>(::sycl::range<2>(kSize, kSize),
                                              ::sycl::range<2>(kSize, kSize)),
            [=](::sycl::nd_item<2> item) {
                const int row = item.get_global_id(0);
                const int col = item.get_global_id(1);
                const int local_row = item.get_local_id(0);
                const int local_col = item.get_local_id(1);

                // Load input coefficients to local memory
                local_tile[local_row][local_col] =
                    static_cast<int>(input[row * kSize + col]);

                item.barrier(::sycl::access::fence_space::local_space);

                // Column transform (inverse)
                int col_data[8];
                for (int r = 0; r < kSize; ++r) {
                    col_data[r] = local_tile[r][local_col];
                }
                IDCT8Kernel::compute(col_data);
                for (int r = 0; r < kSize; ++r) {
                    local_tile[r][local_col] =
                        round_shift(col_data[r], kInvShift8x8_0);
                }

                item.barrier(::sycl::access::fence_space::local_space);

                // Row transform (inverse)
                int row_data[8];
                for (int c = 0; c < kSize; ++c) {
                    row_data[c] = local_tile[local_row][c];
                }
                IDCT8Kernel::compute(row_data);

                // Clamp and store to output
                int val = round_shift(row_data[local_col], kInvShift8x8_1 + kInvShift8x8_2);
                output[row * stride + col] = clamp_pixel(val, bd);
            });
    }).wait();
}

// ============================================================================
// Optimized 4x4 Inverse DCT Implementation
// ============================================================================

void idct4x4_optimized(::sycl::queue& q,
                       const tran_low_t* input,
                       uint16_t* output,
                       int stride,
                       const TxfmParams& params) {
    constexpr int kSize = 4;
    const int bd = params.bd;

    q.submit([&](::sycl::handler& cgh) {
        ::sycl::accessor<int, 2, ::sycl::access::mode::read_write,
                         ::sycl::target::local> local_tile(::sycl::range<2>(kSize, kSize), cgh);

        cgh.parallel_for(::sycl::nd_range<2>(::sycl::range<2>(kSize, kSize),
                                              ::sycl::range<2>(kSize, kSize)),
            [=](::sycl::nd_item<2> item) {
                const int row = item.get_global_id(0);
                const int col = item.get_global_id(1);
                const int local_row = item.get_local_id(0);
                const int local_col = item.get_local_id(1);

                // Load input coefficients
                local_tile[local_row][local_col] =
                    static_cast<int>(input[row * kSize + col]);

                item.barrier(::sycl::access::fence_space::local_space);

                // Column transform
                int col_data[4];
                for (int r = 0; r < kSize; ++r) {
                    col_data[r] = local_tile[r][local_col];
                }
                IDCT4Kernel::compute(col_data);
                for (int r = 0; r < kSize; ++r) {
                    local_tile[r][local_col] = col_data[r];
                }

                item.barrier(::sycl::access::fence_space::local_space);

                // Row transform
                int row_data[4];
                for (int c = 0; c < kSize; ++c) {
                    row_data[c] = local_tile[local_row][c];
                }
                IDCT4Kernel::compute(row_data);

                // Clamp and store
                int val = round_shift(row_data[local_col], 1);
                output[row * stride + col] = clamp_pixel(val, bd);
            });
    }).wait();
}

// ============================================================================
// Batch Processing Implementation
// ============================================================================

void fdct8x8_batch(::sycl::queue& q,
                   const int16_t* const* inputs,
                   tran_low_t* const* outputs,
                   int num_blocks,
                   int stride,
                   const TxfmParams& params) {
    constexpr int kSize = 8;

    // Process multiple blocks in a single kernel launch
    // Each work-group handles one 8x8 block
    q.submit([&](::sycl::handler& cgh) {
        cgh.parallel_for(::sycl::nd_range<2>(
            ::sycl::range<2>(static_cast<size_t>(num_blocks) * kSize, kSize),
            ::sycl::range<2>(kSize, kSize)),
            [=](::sycl::nd_item<2> item) {
                const int block_idx = item.get_global_id(0) / kSize;
                const int row = item.get_local_id(0);
                const int col = item.get_local_id(1);

                if (block_idx >= num_blocks) return;

                const int16_t* input = inputs[block_idx];
                tran_low_t* output = outputs[block_idx];

                // Local memory for this block's tile
                int local_data[8][8];

                // Load from global memory
                local_data[row][col] =
                    static_cast<int>(input[row * stride + col]) << kFwdShift8x8_0;

                item.barrier(::sycl::access::fence_space::local_space);

                // Row transform
                int row_buf[8];
                for (int c = 0; c < 8; ++c) row_buf[c] = local_data[row][c];
                FDCT8Kernel::compute(row_buf);
                for (int c = 0; c < 8; ++c)
                    local_data[row][c] = round_shift(row_buf[c], kFwdShift8x8_1);

                item.barrier(::sycl::access::fence_space::local_space);

                // Column transform
                int col_buf[8];
                for (int r = 0; r < 8; ++r) col_buf[r] = local_data[r][col];
                FDCT8Kernel::compute(col_buf);

                output[row * 8 + col] =
                    static_cast<tran_low_t>(round_shift(col_buf[row], kFwdShift8x8_2));
            });
    }).wait();
}

void fdct4x4_batch(::sycl::queue& q,
                   const int16_t* const* inputs,
                   tran_low_t* const* outputs,
                   int num_blocks,
                   int stride,
                   const TxfmParams& params) {
    constexpr int kSize = 4;

    q.submit([&](::sycl::handler& cgh) {
        cgh.parallel_for(::sycl::nd_range<2>(
            ::sycl::range<2>(static_cast<size_t>(num_blocks) * kSize, kSize),
            ::sycl::range<2>(kSize, kSize)),
            [=](::sycl::nd_item<2> item) {
                const int block_idx = item.get_global_id(0) / kSize;
                const int row = item.get_local_id(0);
                const int col = item.get_local_id(1);

                if (block_idx >= num_blocks) return;

                const int16_t* input = inputs[block_idx];
                tran_low_t* output = outputs[block_idx];

                int local_data[4][4];

                local_data[row][col] = static_cast<int>(input[row * stride + col]);

                item.barrier(::sycl::access::fence_space::local_space);

                int row_buf[4];
                for (int c = 0; c < 4; ++c) row_buf[c] = local_data[row][c];
                FDCT4Kernel::compute(row_buf);
                for (int c = 0; c < 4; ++c) local_data[row][c] = row_buf[c];

                item.barrier(::sycl::access::fence_space::local_space);

                int col_buf[4];
                for (int r = 0; r < 4; ++r) col_buf[r] = local_data[r][col];
                FDCT4Kernel::compute(col_buf);

                output[row * 4 + col] =
                    static_cast<tran_low_t>(round_shift(col_buf[row], 1));
            });
    }).wait();
}

// ============================================================================
// Memory Pool Implementation
// ============================================================================

TxfmMemoryPool::TxfmMemoryPool(::sycl::queue& q, int max_blocks)
    : queue_(q), max_blocks_(max_blocks) {
    // Pre-allocate buffers for each transform size
    int sizes[] = {4, 8, 16, 32};

    for (int i = 0; i < 4; ++i) {
        int size = sizes[i];
        pools_[i].resize(max_blocks);
        available_[i].resize(max_blocks, true);

        for (int j = 0; j < max_blocks; ++j) {
            pools_[i][j].input = ::sycl::malloc_device<int16_t>(size * size, q);
            pools_[i][j].temp = ::sycl::malloc_device<tran_low_t>(size * size, q);
            pools_[i][j].output = ::sycl::malloc_device<tran_low_t>(size * size, q);
        }
    }
}

TxfmMemoryPool::~TxfmMemoryPool() {
    int sizes[] = {4, 8, 16, 32};

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < max_blocks_; ++j) {
            ::sycl::free(pools_[i][j].input, queue_);
            ::sycl::free(pools_[i][j].temp, queue_);
            ::sycl::free(pools_[i][j].output, queue_);
        }
    }
}

TxfmMemoryPool::TxfmBuffers TxfmMemoryPool::acquire(int size) {
    int pool_idx = 0;
    if (size == 4) pool_idx = 0;
    else if (size == 8) pool_idx = 1;
    else if (size == 16) pool_idx = 2;
    else if (size == 32) pool_idx = 3;
    else return {nullptr, nullptr, nullptr};

    for (int i = 0; i < max_blocks_; ++i) {
        if (available_[pool_idx][i]) {
            available_[pool_idx][i] = false;
            return pools_[pool_idx][i];
        }
    }

    // Pool exhausted, allocate new buffer
    TxfmBuffers buf;
    buf.input = ::sycl::malloc_device<int16_t>(size * size, queue_);
    buf.temp = ::sycl::malloc_device<tran_low_t>(size * size, queue_);
    buf.output = ::sycl::malloc_device<tran_low_t>(size * size, queue_);
    return buf;
}

void TxfmMemoryPool::release(const TxfmBuffers& buffers, int size) {
    // Find and mark as available
    int pool_idx = 0;
    if (size == 4) pool_idx = 0;
    else if (size == 8) pool_idx = 1;
    else if (size == 16) pool_idx = 2;
    else if (size == 32) pool_idx = 3;
    else {
        // Allocated outside pool, free directly
        ::sycl::free(buffers.input, queue_);
        ::sycl::free(buffers.temp, queue_);
        ::sycl::free(buffers.output, queue_);
        return;
    }

    for (int i = 0; i < max_blocks_; ++i) {
        if (pools_[pool_idx][i].input == buffers.input) {
            available_[pool_idx][i] = true;
            return;
        }
    }

    // Not found in pool, free directly
    ::sycl::free(buffers.input, queue_);
    ::sycl::free(buffers.temp, queue_);
    ::sycl::free(buffers.output, queue_);
}

// ============================================================================
// Performance Monitoring Functions
// ============================================================================

void set_txfm_profiling_enabled(bool enabled) {
    g_profiling_enabled = enabled;
}

const TxfmPerfStats& get_txfm_perf_stats() {
    return g_last_stats;
}

void TxfmPerfStats::print() const {
    std::cout << "Transform Performance Stats:" << std::endl;
    std::cout << "  Total time: " << total_time_ms << " ms" << std::endl;
    std::cout << "  Avg per block: " << avg_time_per_block_us << " us" << std::endl;
    std::cout << "  Throughput: " << throughput_blocks_per_sec << " blocks/sec" << std::endl;
    std::cout << "  Memory bandwidth: " << memory_bandwidth_gbps << " GB/s" << std::endl;
}

}  // namespace sycl
}  // namespace avm

#endif  // HAVE_SYCL
