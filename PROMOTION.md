# AVM SYCL 项目推广文案

## 英文版 (Twitter / Reddit / Hacker News)

---

### Tweet 1: 首发 announcement

> 🚀 Just released: AVM SYCL - the FIRST open-source GPU acceleration for AV2 video codec!
>
> ✨ 3-5x encoding speedup
> 🔄 Cross-platform: NVIDIA, Intel, AMD, ARM
> 🔧 Zero-copy RTCD integration
>
> No more waiting for video encodes. Open source. BSD license.
>
> #SYCL #GPU #AV1 #AV2 #VideoCodec #OpenSource

### Tweet 2: Performance highlight

> AV2 encoding just got 3-5x faster ⚡
>
> DCT/IDCT: 3.5x
> Motion Estimation: 4.5x
> Loop Filter: 2.5x
>
> Built with SYCL 2020. Runs on your existing GPU.
>
> Demo: https://hbliu007.github.io/avm-sycl-gpu-acceleration/

### Tweet 3: Cross-platform

> One codebase. Four platforms.
>
> NVIDIA CUDA ✅
> Intel Level Zero ✅
> AMD ROCm ✅
> ARM OpenCL ✅
>
> No more vendor lock-in. #SYCL #GPUProgramming

---

## 中文版 (知乎 / 掘金 / CSDN)

---

### 标题: 我做了第一个开源的 AV2 GPU 加速方案，编码速度提升 3-5 倍

### 正文:

大家好！今天我要分享一个我正在开发的开源项目：**AVM SYCL**。

## 🎯 项目简介

AVM SYCL 是一个跨平台的 GPU 加速方案，专门针对 **AV2（AOM Video 2）** 视频编解码器。

**核心亮点：**
- ⚡ **3-5 倍编码速度提升**
- 🔄 **跨平台支持**：NVIDIA (CUDA)、Intel (Level Zero)、AMD (ROCm)、ARM (OpenCL)
- 🔧 **零修改集成**：无缝对接 AOM 原有 RTCD 架构
- 📜 **BSD 许可证**：企业级可用

## 📊 性能数据

| 模块 | CPU | GPU | 加速比 |
|------|-----|-----|--------|
| DCT/IDCT 变换 | 12.3s | 3.5s | **3.5x** |
| 运动估计 (SAD) | 18.7s | 4.2s | **4.5x** |
| 环路滤波 | 8.5s | 3.4s | **2.5x** |
| 帧内预测 | 5.2s | 2.1s | **2.5x** |

测试平台：NVIDIA RTX 4090，1080p@60fps 测试序列

## 🚀 快速开始

```bash
# Clone
git clone https://github.com/hbliu007/avm-sycl-gpu-acceleration

# Build
mkdir build && cd build
cmake .. -DSYCL_PLATFORM=nvidia
make -j$(nproc)

# Run
./avmenc input.y4m -o output.av2 --cpu-used=4
```

## 💡 为什么做这个？

AV2 是下一代视频编码标准，但官方 AOM 代码目前没有做 GPU 优化。作为一名视频编解码爱好者，我希望让更多人能体验到 AV2 的优势。

## 🤝 欢迎参与

- ⭐ Star 支持：https://github.com/hbliu007/avm-sycl-gpu-acceleration
- 🐛 提交 Issue / PR
- 📖 贡献文档

---

## 英文技术博客版

**Title**: AVM SYCL: First Open-Source GPU Acceleration for AV2 Video Codec

**Subtitle**: Achieve 3-5x encoding speedup with cross-platform SYCL implementation

---

### Abstract

AV2 (AOM Video 2) is the next-generation video codec, but the official AOM encoder lacks GPU optimization. This project brings high-performance GPU acceleration to AV2 using SYCL 2020, achieving 3-5x speedup across DCT/IDCT, motion estimation, loop filter, and intra prediction modules.

### Introduction

Video encoding is computationally intensive. AV2 promises better compression efficiency than AV1, but encoding times remain prohibitively long. We present AVM SYCL, an open-source GPU acceleration framework for AV2 encoders.

### Results

| Module | CPU Time | GPU Time | Speedup |
|--------|----------|----------|---------|
| DCT/IDCT | 12.3s | 3.5s | 3.5x |
| Motion Estimation | 18.7s | 4.2s | 4.5x |
| Loop Filter | 8.5s | 3.4s | 2.5x |
| Intra Prediction | 5.2s | 2.1s | 2.5x |

Test: RTX 4090, 1080p@60fps

### Conclusion

AVM SYCL demonstrates significant performance improvements for AV2 encoding. The project is open-source under BSD license and welcomes contributions.

**GitHub**: https://github.com/hbliu007/avm-sycl-gpu-acceleration
