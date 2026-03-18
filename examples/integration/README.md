# Integration Examples

This directory contains examples for integrating SYCL GPU acceleration with popular frameworks.

## Available Examples

### FFmpeg Integration (`ffmpeg_integration.cpp`)

Demonstrates how to use SYCL kernels in FFmpeg video encoding/decoding pipelines.

**Features:**
- GPU-accelerated DCT for frame compression
- Motion estimation with SAD computation
- Frame-level processing pipeline

**Requirements:**
```bash
# Ubuntu
sudo apt install libavcodec-dev libavutil-dev

# Build
g++ -std=c++17 ffmpeg_integration.cpp -lavcodec -lavutil -o ffmpeg_integration
```

### OpenCV Integration (`opencv_integration.cpp`)

Demonstrates how to use SYCL kernels with OpenCV for real-time video processing.

**Features:**
- Block-based DCT for image compression
- Motion vector computation for optical flow
- Motion visualization overlay

**Requirements:**
```bash
# Ubuntu
sudo apt install libopencv-dev

# Build
g++ -std=c++17 opencv_integration.cpp `pkg-config --cflags --libs opencv4` -o opencv_integration
```

## Common Patterns

### 1. Frame Processing Loop

```cpp
auto& ctx = avm::sycl::SYCLContext::instance();
ctx.initialize();

while (has_more_frames()) {
    auto frame = get_next_frame();

    // GPU DCT
    avm::sycl::fdct8x8(ctx.queue(), frame.data(), dct_output);

    // GPU Motion Estimation
    auto sad = avm::sycl::sad16x16(ctx.queue(), ref, cur);

    ctx.queue().wait();  // Synchronize if needed
}
```

### 2. Batch Processing

```cpp
// Process multiple blocks in parallel
std::vector<sycl::event> events;

for (int i = 0; i < num_blocks; ++i) {
    auto evt = avm::sycl::fdct8x8_async(ctx.queue(), blocks[i], outputs[i]);
    events.push_back(evt);
}

// Wait for all
sycl::wait_for(events);
```

### 3. Memory Management

```cpp
// Use USM for efficient GPU memory
auto buffer = sycl::malloc_device<uint8_t>(size, ctx.queue());

// ... use buffer ...

// Don't forget to free
sycl::free(buffer, ctx.queue());
```

## Performance Tips

1. **Batch operations**: Process multiple blocks together to amortize kernel launch overhead
2. **USM allocation**: Use SYCL USM for frequent data transfers
3. **Async execution**: Use async kernel variants and wait only when needed
4. **Device selection**: Let SYCLContext choose the best device automatically

## Troubleshooting

### "SYCL not available"
- Ensure SYCL runtime is installed (Intel oneAPI or AdaptiveCpp)
- Check that `HAVE_SYCL` is defined during compilation

### Poor performance
- Verify GPU is being used: `ctx.is_gpu()` should return `true`
- Check compute units: `ctx.compute_units()` should be > 1
- Profile with: `SYCL_PI_TRACE=1 ./your_program`

### Memory errors
- Ensure buffers are properly sized
- Check for queue synchronization issues
- Verify USM pointers are valid before use
