# SYCL GPU Acceleration Architecture

## Overview

This document describes the architecture of the SYCL GPU acceleration module for AV2 codec.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AV2 Encoder Application                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RTCD (Runtime CPU Dispatch)                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  CPU Path   │  │  SIMD Path  │  │     SYCL GPU Path       │  │
│  │  (C/C++)    │  │ (NEON/SSE)  │  │  (CUDA/Level-Zero/CL)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SYCL Wrapper Layer                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ should_use_sycl() → Device Check + Heuristic Decision   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SYCL Context Manager                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Device Enum  │  │ Device Score │  │ Queue Management     │   │
│  │ & Selection  │  │ & Ranking    │  │ & Error Handling     │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SYCL Compute Kernels                          │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌─────────────┐   │
│  │ Transform  │ │  Motion    │ │   Loop     │ │   Intra     │   │
│  │ DCT/IDCT   │ │ Estimation │ │  Filter    │ │ Prediction  │   │
│  └────────────┘ └────────────┘ └────────────┘ └─────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. SYCL Context (sycl_context.hpp/cpp)

Singleton class managing SYCL device and queue:

```cpp
class SYCLContext {
    // Device selection with scoring
    int score_device(const sycl::device& dev);

    // Automatic best device selection
    sycl::device select_best_device();

    // Queue for kernel submission
    sycl::queue queue_;
};
```

**Device Scoring Algorithm:**

| Factor | Points |
|--------|--------|
| is_gpu() | +1000 |
| Vendor: NVIDIA | +300 |
| Vendor: Intel | +200 |
| Vendor: Apple | +100 |
| Vendor: AMD | +50 |
| Memory > 4GB | +100 |
| Each compute unit | +10 |

### 2. Transform Kernels (sycl_txfm.hpp/cpp)

Implements forward and inverse DCT transforms:

**Supported Transforms:**
- `fdct8x8` - Forward 8x8 DCT
- `idct8x8` - Inverse 8x8 DCT
- `fdct16x16` - Forward 16x16 DCT
- `idct16x16` - Inverse 16x16 DCT
- `fdct32x32` - Forward 32x32 DCT
- `idct32x32` - Inverse 32x32 DCT

**Algorithm:** Two-pass row-column decomposition

```
Input Block (8x8)
      │
      ▼
┌─────────────┐
│ Row 1D DCT  │ ← 8 parallel work-items
├─────────────┤
│ Row 1D DCT  │
├─────────────┤
│    ...      │
├─────────────┤
│ Row 1D DCT  │
└─────────────┘
      │
      ▼
┌─────────────┐
│ Col 1D DCT  │ ← 8 parallel work-items
├─────────────┤
│ Col 1D DCT  │
├─────────────┤
│    ...      │
├─────────────┤
│ Col 1D DCT  │
└─────────────┘
      │
      ▼
Output Block (8x8)
```

### 3. Motion Estimation (sycl_me.hpp/cpp)

Implements SAD (Sum of Absolute Differences) for block matching:

**Supported SAD Functions:**
- `sad4x4` - 4x4 block SAD
- `sad8x8` - 8x8 block SAD
- `sad16x16` - 16x16 block SAD
- `sad32x32` - 32x32 block SAD
- `sad64x64` - 64x64 block SAD

**Algorithm:** Parallel reduction with work-groups

```
┌────────────────────────────────┐
│  Block Comparison (16x16)      │
│  ┌────┬────┬────┬────┐         │
│  │ S1 │ S2 │ S3 │ S4 │         │
│  ├────┼────┼────┼────┤         │
│  │ S5 │ S6 │ S7 │ S8 │  ...    │
│  ├────┼────┼────┼────┤         │
│  │    │    │    │    │         │
│  └────┴────┴────┴────┘         │
└────────────────────────────────┘
           │
           ▼
┌────────────────────────────────┐
│  Work-Group Reduction          │
│  subgroup_reduce(add)          │
└────────────────────────────────┘
           │
           ▼
       Final SAD
```

### 4. Loop Filter (sycl_lpf.hpp/cpp)

Implements deblocking filter for artifact removal:

**Filter Types:**
- Vertical edge filter
- Horizontal edge filter
- Luma filter
- Chroma filter

### 5. Intra Prediction (sycl_intra.hpp/cpp)

Implements intra-frame prediction modes:

**Prediction Modes:**
- DC prediction (average value)
- H prediction (horizontal)
- V prediction (vertical)
- Smooth prediction
- Directional predictions

## Memory Model

```
┌─────────────────────────────────────────────────────┐
│                    Host Memory                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ Input Data  │  │ Output Data │  │ Parameters  │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
         │                    ▲              │
         │ memcpy             │ memcpy       │ USM
         ▼                    │              ▼
┌─────────────────────────────────────────────────────┐
│                    Device Memory                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ USM Buffer  │  │ USM Buffer  │  │ Local Mem   │  │
│  │ (input)     │  │ (output)    │  │ (shared)    │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Integration with AV2

The SYCL module integrates via RTCD mechanism:

```cpp
// In avm_dsp/rtcd.c

#if HAVE_SYCL
#include "sycl/sycl_wrapper.hpp"

void avm_fdct8x8_sycl(const int16_t *input, tran_low_t *output) {
    if (avm::sycl::should_use_sycl()) {
        auto& ctx = avm::sycl::SYCLContext::instance();
        avm::sycl::fdct8x8(ctx.queue(), input, output);
    } else {
        // Fallback to CPU
        avm_fdct8x8_c(input, output);
    }
}
#endif
```

## Performance Considerations

### Kernel Launch Overhead

- Minimum worthwhile work: ~10,000 operations
- Batch small operations when possible
- Use USM for frequent transfers

### Memory Bandwidth

- DCT: ~80% memory bound
- SAD: ~60% compute bound
- Loop Filter: ~70% memory bound

### Recommended Usage

| Frame Size | Recommended Path |
|------------|-----------------|
| < 480p | CPU SIMD |
| 480p - 1080p | GPU (SYCL) |
| > 1080p | GPU (SYCL) + Tiling |

## Future Work

1. **Tile-based processing** for 4K+ frames
2. **Async execution** with double buffering
3. **Multi-GPU support** for scaling
4. **Dynamic load balancing** between CPU/GPU
