# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-18

### Added
- **Transform Kernels**: DCT/IDCT for 8x8, 16x16, 32x32 blocks
- **Motion Estimation**: SAD computation for 4x4, 8x8, 16x16, 32x32, 64x64 blocks
- **Loop Filter**: Edge-adaptive deblocking filter kernels
- **Intra Prediction**: DC, Horizontal, Vertical prediction modes
- **Device Management**: Automatic GPU selection with intelligent scoring
- **SYCL Context**: Singleton pattern for queue and device management
- **CMake Integration**: SYCL detection and configuration module
- **Unit Tests**: Context, transform, and performance tests
- **Examples**: Basic usage demonstration

### Platform Support
- NVIDIA GPUs via CUDA backend
- Intel GPUs via Level Zero backend
- AMD GPUs via HIP backend (experimental)
- CPUs via OpenCL backend (fallback)

### Documentation
- Architecture documentation
- API reference in headers
- Contributing guidelines

## [Unreleased]

### Planned
- Python bindings
- FFmpeg integration
- OpenCV integration
- Docker images
- vcpkg/conan packages
- Performance benchmarks dashboard

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2026-03-18 | Initial release with core kernels |

---

[1.0.0]: https://github.com/hbliu007/avm-sycl-gpu-acceleration/releases/tag/v1.0.0
