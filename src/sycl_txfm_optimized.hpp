/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved
 *
 * Optimized SYCL transform kernels for AVM encoder
 * Addresses performance bottlenecks in original implementation:
 * - Single kernel launch for 2D transforms
 * - Local memory tiling for reduced global memory access
 * - Proper work-group utilization for both row and column passes
 * - Batch processing support for multiple blocks
 */

#ifndef AVM_AVM_DSP_SYCL_SYCL_TXFM_OPTIMIZED_HPP_
#define AVM_AVM_DSP_SYCL_SYCL_TXFM_OPTIMIZED_HPP_

#include <sycl/sycl.hpp>
#include "av2/common/enums.h"
#include "avm_dsp_common.h"

namespace avm {
namespace sycl {

// Transform parameters
struct TxfmParams {
  TX_SIZE tx_size;
  TX_TYPE tx_type;
  int bd;              // Bit depth
  int eob;             // End of block (for inverse)
};

// Batch processing parameters
struct BatchTxfmParams {
  const void* inputs;   // Pointer to array of input pointers
  void* outputs;        // Pointer to array of output pointers
  int batch_size;
  int stride;
};

// ============================================================================
// Optimized Forward DCT with Local Memory Tiling
// ============================================================================

/// @brief Optimized 2D 8x8 DCT with single kernel launch
/// Uses local memory tile to process entire 8x8 block in one work-group
void fdct8x8_optimized(::sycl::queue& q,
                       const int16_t* input,
                       tran_low_t* output,
                       int stride,
                       const TxfmParams& params);

/// @brief Optimized 2D 4x4 DCT with single kernel launch
void fdct4x4_optimized(::sycl::queue& q,
                       const int16_t* input,
                       tran_low_t* output,
                       int stride,
                       const TxfmParams& params);

// ============================================================================
// Optimized Inverse DCT with Local Memory Tiling
// ============================================================================

/// @brief Optimized 2D 8x8 IDCT with single kernel launch
void idct8x8_optimized(::sycl::queue& q,
                       const tran_low_t* input,
                       uint16_t* output,
                       int stride,
                       const TxfmParams& params);

/// @brief Optimized 2D 4x4 IDCT with single kernel launch
void idct4x4_optimized(::sycl::queue& q,
                       const tran_low_t* input,
                       uint16_t* output,
                       int stride,
                       const TxfmParams& params);

// ============================================================================
// Batch Processing for Multiple Transforms
// ============================================================================

/// @brief Batch process multiple 8x8 DCT transforms
/// Reduces kernel launch overhead by processing multiple blocks per kernel
void fdct8x8_batch(::sycl::queue& q,
                   const int16_t* const* inputs,
                   tran_low_t* const* outputs,
                   int num_blocks,
                   int stride,
                   const TxfmParams& params);

/// @brief Batch process multiple 4x4 DCT transforms
void fdct4x4_batch(::sycl::queue& q,
                   const int16_t* const* inputs,
                   tran_low_t* const* outputs,
                   int num_blocks,
                   int stride,
                   const TxfmParams& params);

// ============================================================================
// Persistent Memory Allocator for Reduced Allocation Overhead
// ============================================================================

/// @brief Memory pool for transform operations
/// Pre-allocates device memory to avoid per-transform allocation overhead
class TxfmMemoryPool {
public:
    explicit TxfmMemoryPool(::sycl::queue& q, int max_blocks = 1024);
    ~TxfmMemoryPool();

    // Get pre-allocated device memory for a transform
    struct TxfmBuffers {
        int16_t* input;
        tran_low_t* temp;
        tran_low_t* output;
    };

    TxfmBuffers acquire(int size);
    void release(const TxfmBuffers& buffers, int size);

private:
    ::sycl::queue& queue_;
    std::vector<TxfmBuffers> pools_[4];  // Pools for 4, 8, 16, 32
    std::vector<bool> available_[4];
    int max_blocks_;
};

// ============================================================================
// Performance Monitoring
// ============================================================================

struct TxfmPerfStats {
    double total_time_ms;
    double avg_time_per_block_us;
    double throughput_blocks_per_sec;
    double memory_bandwidth_gbps;

    void print() const;
};

/// @brief Enable performance profiling for transform operations
void set_txfm_profiling_enabled(bool enabled);

/// @brief Get performance statistics from last transform operation
const TxfmPerfStats& get_txfm_perf_stats();

}  // namespace sycl
}  // namespace avm

#endif  // AVM_AVM_DSP_SYCL_SYCL_TXFM_OPTIMIZED_HPP_
