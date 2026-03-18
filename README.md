# AVM SYCL GPU Acceleration

[![CI](https://github.com/hbliu007/avm-sycl-gpu-acceleration/actions/workflows/ci.yml/badge.svg)](https://github.com/hbliu007/avm-sycl-gpu-acceleration/actions/workflows/ci.yml)
[![Security](https://github.com/hbliu007/avm-sycl-gpu-acceleration/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/hbliu007/avm-sycl-gpu-acceleration/security)
[![License](https://img.shields.io/badge/License-BSD%203--Clause%20Clear-blue.svg)](https://opensource.org/licenses/BSD-3-Clause-Clear)
[![SYCL](https://img.shields.io/badge/SYCL-2020-purple.svg)](https://www.khronos.org/sycl/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-orange.svg)](https://en.cppreference.com/w/cpp/17)
[![GitHub release](https://img.shields.io/github/v/release/hbliu007/avm-sycl-gpu-acceleration?include_prereleases)](https://github.com/hbliu007/avm-sycl-gpu-acceleration/releases)

> Cross-platform SYCL GPU acceleration for AV2 (AOM Video 2) codec

## Features

- **Cross-Platform GPU Support**: Write once, run on any GPU
- **Zero-Copy Integration**: Seamless RTCD (Runtime CPU Dispatch) integration
- **Automatic Device Selection**: Intelligent GPU scoring and selection
- **Fallback Mechanism**: Automatic CPU fallback when GPU unavailable
- **Production Ready**: Comprehensive unit tests and performance benchmarks

### Performance Target: 3-5x Encoding Speedup

| Module | CPU (NEON/SSE) | GPU (SYCL) | Speedup |
|--------|---------------|------------|---------|
| Transform (DCT/IDCT) | 1.0x | 3-4x | 3-4x |
| Motion Estimation (SAD) | 1.0x | 4-5x | 4-5x |
| Loop Filter | 1.0x | 2-3x | 2-3x |
| Intra Prediction | 1.0x | 2-3x | 2-3x |

## Platform Support Matrix

### GPU Hardware Support

| Vendor | Architecture | SYCL Backend | DCT/IDCT | SAD | Loop Filter | Intra |
|--------|-------------|--------------|:--------:|:---:|:-----------:|:-----:|
| **NVIDIA** | Ampere (RTX 30xx) | CUDA | ✅ | ✅ | ✅ | ✅ |
| **NVIDIA** | Ada Lovelace (RTX 40xx) | CUDA | ✅ | ✅ | ✅ | ✅ |
| **NVIDIA** | Turing (RTX 20xx) | CUDA | ✅ | ✅ | ✅ | ✅ |
| **Intel** | Arc A-Series | Level Zero | ✅ | ✅ | ✅ | ✅ |
| **Intel** | Xe Integrated | Level Zero | ✅ | ✅ | ✅ | ✅ |
| **Intel** | HD Graphics | OpenCL | ✅ | ✅ | ⚠️ | ⚠️ |
| **AMD** | RDNA2/3 | HIP | 🔄 | 🔄 | 🔄 | 🔄 |
| **ARM** | Mali | OpenCL | 🔄 | 🔄 | 🔄 | 🔄 |
| **Apple** | M-Series | Metal* | ⚠️ | ⚠️ | ⚠️ | ⚠️ |

Legend: ✅ Supported | 🔄 Experimental | ⚠️ Limited | ❌ Not Supported

*Apple Silicon via OpenCL (Metal backend not yet available in SYCL)

### Compiler Support

| Compiler | Version | Linux | Windows | macOS |
|----------|---------|:-----:|:-------:|:-----:|
| Intel DPC++ (icpx) | 2024.0+ | ✅ | ✅ | ❌ |
| AdaptiveCpp | 23.10+ | ✅ | 🔄 | ⚠️ CPU only |
| clang++ + DPC++ | 16+ | ✅ | ✅ | ⚠️ CPU only |

### Operating System Support

| OS | Version | Status |
|----|---------|--------|
| Ubuntu | 22.04 LTS | ✅ Primary |
| Ubuntu | 20.04 LTS | ✅ Supported |
| Windows | 10/11 | ✅ Supported |
| macOS | 13+ | ⚠️ CPU only |
| CentOS/RHEL | 8+ | ✅ Supported |

## Quick Start

### Prerequisites

- **Compiler**: Intel DPC++ (icpx) or AdaptiveCpp (hipSYCL)
- **CMake**: 3.20+
- **SYCL Runtime**: One of:
  - Intel oneAPI DPC++ Runtime
  - AdaptiveCpp with CUDA/ROCm/OpenCL backend

### Installation

#### From Source (Linux)

```bash
# Install Intel oneAPI (recommended)
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
  gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
  sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update && sudo apt install intel-oneapi-dpcpp-cpp

# Build
git clone https://github.com/hbliu007/avm-sycl-gpu-acceleration.git
cd avm-sycl-gpu-acceleration
source /opt/intel/oneapi/setvars.sh
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

#### From Source (Windows)

```powershell
# Install Intel oneAPI from https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html

# Build
git clone https://github.com/hbliu007/avm-sycl-gpu-acceleration.git
cd avm-sycl-gpu-acceleration
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -DCMAKE_CXX_COMPILER=icpx
cmake --build . --config Release
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
├── examples/
│   ├── basic_usage.cpp       # Basic integration example
│   └── integration/          # Framework integration examples
├── docs/
│   └── architecture.md       # Detailed architecture docs
├── .github/
│   ├── workflows/ci.yml      # CI/CD pipeline
│   └── ISSUE_TEMPLATE/       # Issue templates
├── LICENSE
├── CONTRIBUTING.md
├── CHANGELOG.md
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
    uint32_t sad = avm::sycl::sad16x16(ctx.queue(), ref, candidate);
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
```

## Device Selection Priority

| Device Type | Base Score | Bonus |
|-------------|-----------|-------|
| GPU | 1000 | + |
| NVIDIA | - | +300 |
| Intel | - | +200 |
| Apple | - | +100 |
| AMD | - | +50 |
| High Memory (>4GB) | - | +100 |
| Each compute unit | - | +10 |

## Documentation

- [Architecture Guide](docs/architecture.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Steps

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/avm-sycl-gpu-acceleration.git

# Create feature branch
git checkout -b feature/your-feature

# Build and test
mkdir build && cd build
cmake .. -DAVM_BUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure

# Submit PR
```

## License

This project is licensed under the BSD 3-Clause Clear License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [AOMedia](https://aomedia.org/) for the AV2 codec reference implementation
- [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) for DPC++ compiler
- [Khronos Group](https://www.khronos.org/sycl/) for SYCL specification
- [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) for portable SYCL implementation

## Citation

If you use this project in your research, please cite:

```bibtex
@software{avm_sycl_gpu,
  title = {AVM SYCL GPU Acceleration},
  author = {Liu, Hongbo},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/hbliu007/avm-sycl-gpu-acceleration}
}
```

---

**⭐ Star this repo if you find it useful!**
