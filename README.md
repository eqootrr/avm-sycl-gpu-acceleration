# AVM SYCL GPU Acceleration

[![License](https://img.shields.io/badge/License-BSD%203--Clause%20Clear-blue.svg)](https://opensource.org/licenses/BSD-3-Clause-Clear)
[![SYCL](https://img.shields.io/badge/SYCL-2020-purple.svg)](https://www.khronos.org/sycl/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-orange.svg)](https://en.cppreference.com/w/cpp/17)
[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20%7C%20Intel%20%7C%20AMD-green.svg)]()

> Cross-platform SYCL GPU acceleration for AV2 (AOM Video 2) codec

## Overview

This project implements GPU-accelerated kernels for the AV2 video codec using SYCL (SYCL for OpenCL), enabling cross-platform GPU acceleration on NVIDIA, Intel, AMD, and ARM GPUs.

### Performance Target: 3-5x Encoding Speedup

| Module | CPU (NEON/SSE) | GPU (SYCL) | Speedup |
|--------|---------------|------------|---------|
| Transform (DCT/IDCT) | 1.0x | 3-4x | 3-4x |
| Motion Estimation (SAD) | 1.0x | 4-5x | 4-5x |
| Loop Filter | 1.0x | 2-3x | 2-3x |
| Intra Prediction | 1.0x | 2-3x | 2-3x |

## Features

- **Cross-Platform GPU Support**: NVIDIA (CUDA), Intel (Level Zero), AMD (ROCm), ARM (OpenCL)
- **Zero-Copy Integration**: Seamless RTCD (Runtime CPU Dispatch) integration
- **Automatic Device Selection**: Intelligent GPU scoring and selection
- **Fallback Mechanism**: Automatic CPU fallback when GPU unavailable
- **Production Ready**: Comprehensive unit tests and performance benchmarks

## Architecture

```
+------------------+     +------------------+     +------------------+
|   AV2 Encoder    | --> |   SYCL Wrapper   | --> |   GPU Kernels    |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
+------------------+     +------------------+     +------------------+
|  RTCD Dispatch   |     |  Device Manager  |     |  DCT/IDCT/SAD    |
+------------------+     +------------------+     +------------------+
                                                         |
                                                         v
                                                 +------------------+
                                                 |  Loop Filter /   |
                                                 |  Intra Predict   |
                                                 +------------------+
```

## Supported Platforms

| Platform | Backend | Status |
|----------|---------|--------|
| NVIDIA GPU | CUDA | Supported |
| Intel GPU | Level Zero | Supported |
| Intel CPU | OpenCL | Supported |
| AMD GPU | ROCm/HIP | Supported |
| Apple Silicon | Metal (via OpenCL) | Limited |

## Quick Start

### Prerequisites

- **Compiler**: Intel DPC++ (icpx) or AdaptiveCpp (hipSYCL)
- **CMake**: 3.20+
- **SYCL Runtime**: One of:
  - Intel oneAPI DPC++ Runtime
  - AdaptiveCpp with CUDA/ROCm/OpenCL backend

### Build

```bash
# Clone repository
git clone https://github.com/hbliu007/avm-sycl-gpu-acceleration.git
cd avm-sycl-gpu-acceleration

# Build with Intel DPC++
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Or with AdaptiveCpp
cmake .. -DCMAKE_CXX_COMPILER=clang++ \
         -DSYCL_COMPILER=hipSYCL \
         -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Run Tests

```bash
# Unit tests
./tests/sycl_context_test
./tests/sycl_txfm_test

# Performance benchmarks
./tests/sycl_perf_test --benchmark --frames=100
```

## Project Structure

```
avm-sycl-gpu-acceleration/
├── src/
│   ├── sycl_context.hpp      # Device management
│   ├── sycl_context.cpp      # Device selection logic
│   ├── sycl_txfm.hpp/cpp     # DCT/IDCT transforms
│   ├── sycl_me.hpp/cpp       # Motion estimation (SAD)
│   ├── sycl_lpf.hpp/cpp      # Loop filter
│   ├── sycl_intra.hpp/cpp    # Intra prediction
│   ├── sycl_wrapper.hpp      # Unified API wrapper
│   └── CMakeLists.txt        # Build configuration
├── cmake/
│   └── sycl.cmake            # SYCL detection & configuration
├── tests/
│   ├── sycl_context_test.cpp # Device management tests
│   ├── sycl_txfm_test.cpp    # Transform accuracy tests
│   └── sycl_perf_test.cpp    # Performance benchmarks
├── docs/
│   └── architecture.md       # Detailed architecture docs
├── examples/
│   └── basic_usage.cpp       # Integration example
├── LICENSE
└── README.md
```

## API Usage

### Basic Integration

```cpp
#include "sycl_wrapper.hpp"

// Check if SYCL GPU is available
if (avm::sycl::should_use_sycl()) {
    // Initialize SYCL context
    auto& ctx = avm::sycl::SYCLContext::instance();
    ctx.initialize();

    // Use GPU-accelerated DCT
    avm::sycl::fdct8x8(ctx.queue(), input, output);

    // Use GPU-accelerated SAD
    int sad = avm::sycl::sad4x4(ctx.queue(), ref, candidate);
}
```

### Device Selection

```cpp
// List available devices
auto devices = avm::sycl::SYCLContext::instance().list_devices();
for (const auto& dev : devices) {
    std::cout << dev.name << " (" << dev.vendor << ")"
              << " GPU: " << dev.is_gpu
              << " Compute Units: " << dev.compute_units << std::endl;
}

// Check current device
auto& ctx = avm::sycl::SYCLContext::instance();
std::cout << "Using: " << ctx.backend_name()
          << " (GPU: " << ctx.is_gpu() << ")"
          << " Compute Units: " << ctx.compute_units() << std::endl;
```

## Device Selection Priority

The device selector uses a scoring algorithm:

| Device Type | Base Score | Bonus |
|-------------|-----------|-------|
| GPU | 1000 | + |
| NVIDIA | - | +300 |
| Intel | - | +200 |
| Apple | - | +100 |
| AMD | - | +50 |
| High Memory | - | +100 |
| Many Compute Units | - | +10 per CU |

## Kernel Implementation Details

### DCT 8x8 Transform

Two-pass row-column decomposition with butterfly algorithm:
- Stage 1: Row-wise 1D DCT (8 work-items per row)
- Stage 2: Column-wise 1D DCT
- Precision: 16-bit fixed-point with 12-bit shift

### Motion Estimation SAD

Parallel reduction with work-group optimization:
- 4x4: 16 work-items, single reduction
- 16x16: 256 work-items, hierarchical reduction
- Memory: Local memory for intermediate results

### Loop Filter

Edge-adaptive deblocking filter:
- Parallel edge processing
- Shared memory for neighbor samples
- Conditional filtering based on boundary strength

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install dependencies (Ubuntu)
sudo apt install intel-oneapi-dpcpp-cpp cmake

# Build with debug symbols
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTS=ON
make -j$(nproc)

# Run all tests
ctest --output-on-failure
```

## License

This project is licensed under the BSD 3-Clause Clear License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [AOMedia](https://aomedia.org/) for the AV2 codec reference implementation
- [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) for DPC++ compiler
- [Khronos Group](https://www.khronos.org/sycl/) for SYCL specification

## Citation

If you use this project in your research, please cite:

```bibtex
@software{avm_sycl_gpu,
  title = {AVM SYCL GPU Acceleration},
  author = {Liu, Hongbo},
  year = {2026},
  url = {https://github.com/hbliu007/avm-sycl-gpu-acceleration}
}
```

---

**Star this repo if you find it useful!**
