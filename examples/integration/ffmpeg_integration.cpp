// examples/integration/ffmpeg_integration.cpp
// FFmpeg integration example for SYCL GPU acceleration

/*
 * This example demonstrates how to integrate SYCL GPU acceleration
 * with FFmpeg for video encoding/decoding pipelines.
 *
 * Prerequisites:
 * - FFmpeg development libraries (libavcodec, libavutil)
 * - SYCL runtime (Intel DPC++ or AdaptiveCpp)
 *
 * Build:
 *   g++ -std=c++17 ffmpeg_integration.cpp -lavcodec -lavutil -o ffmpeg_integration
 *   # Or with CMake:
 *   cmake .. -DAVM_BUILD_EXAMPLES=ON
 */

#include <iostream>
#include <vector>
#include <memory>

// SYCL headers
#ifdef HAVE_SYCL
#include "sycl_context.hpp"
#include "sycl_txfm.hpp"
#include "sycl_me.hpp"
#include "sycl_wrapper.hpp"
#endif

// FFmpeg headers (commented out for standalone compilation)
// extern "C" {
// #include <libavcodec/avcodec.h>
// #include <libavutil/frame.h>
// #include <libavutil/imgutils.h>
// }

namespace avm {
namespace ffmpeg {

/**
 * @brief SYCL-accelerated video frame processor
 *
 * This class wraps SYCL GPU kernels for use in FFmpeg pipelines.
 * It handles frame format conversion and memory management.
 */
class SYCLFrameProcessor {
public:
    SYCLFrameProcessor(int width, int height)
        : width_(width), height_(height), initialized_(false) {}

    ~SYCLFrameProcessor() = default;

    /**
     * @brief Initialize SYCL context and allocate resources
     * @return true if initialization successful
     */
    bool initialize() {
#ifdef HAVE_SYCL
        if (!avm::sycl::should_use_sycl()) {
            std::cout << "SYCL not available, using CPU fallback" << std::endl;
            return false;
        }

        auto& ctx = avm::sycl::SYCLContext::instance();
        if (!ctx.initialize()) {
            std::cerr << "Failed to initialize SYCL context" << std::endl;
            return false;
        }

        std::cout << "SYCL initialized on: " << ctx.backend_name() << std::endl;
        std::cout << "GPU: " << (ctx.is_gpu() ? "Yes" : "No") << std::endl;
        std::cout << "Compute Units: " << ctx.compute_units() << std::endl;

        // Allocate SYCL USM buffers
        size_t frame_size = width_ * height_;
        sycl_buffer_ = sycl::malloc_device<uint8_t>(frame_size * 3 / 2, ctx.queue());

        initialized_ = true;
        return true;
#else
        std::cout << "SYCL support not compiled in" << std::endl;
        return false;
#endif
    }

    /**
     * @brief Process frame with GPU-accelerated DCT
     * @param frame_data Input frame data (YUV420P)
     * @param output DCT coefficients output
     */
    void process_dct(const uint8_t* frame_data, int32_t* output) {
#ifdef HAVE_SYCL
        if (!initialized_) return;

        auto& ctx = avm::sycl::SYCLContext::instance();
        auto& q = ctx.queue();

        // Process 8x8 blocks
        int blocks_x = width_ / 8;
        int blocks_y = height_ / 8;

        for (int by = 0; by < blocks_y; ++by) {
            for (int bx = 0; bx < blocks_x; ++bx) {
                // Extract 8x8 block and compute DCT
                std::vector<int16_t> block(64);
                for (int y = 0; y < 8; ++y) {
                    for (int x = 0; x < 8; ++x) {
                        int src_idx = (by * 8 + y) * width_ + (bx * 8 + x);
                        block[y * 8 + x] = static_cast<int16_t>(frame_data[src_idx]) - 128;
                    }
                }

                // GPU DCT
                avm::sycl::fdct8x8(q, block.data(), output + (by * blocks_x + bx) * 64);
            }
        }

        q.wait();  // Synchronize
#endif
    }

    /**
     * @brief Compute motion estimation between two frames
     * @param ref_frame Reference frame
     * @param cur_frame Current frame
     * @param mv_x Motion vector X output
     * @param mv_y Motion vector Y output
     * @param sad SAD value output
     */
    void motion_estimation(const uint8_t* ref_frame, const uint8_t* cur_frame,
                           int* mv_x, int* mv_y, uint32_t* sad) {
#ifdef HAVE_SYCL
        if (!initialized_) return;

        auto& ctx = avm::sycl::SYCLContext::instance();
        auto& q = ctx.queue();

        // 16x16 block motion estimation
        int blocks_x = width_ / 16;
        int blocks_y = height_ / 16;

        for (int by = 0; by < blocks_y; ++by) {
            for (int bx = 0; bx < blocks_x; ++bx) {
                uint32_t best_sad = UINT32_MAX;
                int best_mv_x = 0, best_mv_y = 0;

                // Search range: -16 to +16
                for (int dy = -16; dy <= 16; dy += 2) {
                    for (int dx = -16; dx <= 16; dx += 2) {
                        int ref_x = bx * 16 + dx;
                        int ref_y = by * 16 + dy;

                        // Boundary check
                        if (ref_x < 0 || ref_x + 16 > width_) continue;
                        if (ref_y < 0 || ref_y + 16 > height_) continue;

                        // Extract blocks and compute SAD
                        std::vector<uint8_t> ref_block(256), cur_block(256);
                        for (int y = 0; y < 16; ++y) {
                            for (int x = 0; x < 16; ++x) {
                                ref_block[y * 16 + x] = ref_frame[(ref_y + y) * width_ + (ref_x + x)];
                                cur_block[y * 16 + x] = cur_frame[(by * 16 + y) * width_ + (bx * 16 + x)];
                            }
                        }

                        uint32_t current_sad = avm::sycl::sad16x16(q, ref_block.data(), cur_block.data());

                        if (current_sad < best_sad) {
                            best_sad = current_sad;
                            best_mv_x = dx;
                            best_mv_y = dy;
                        }
                    }
                }

                int block_idx = by * blocks_x + bx;
                mv_x[block_idx] = best_mv_x;
                mv_y[block_idx] = best_mv_y;
                sad[block_idx] = best_sad;
            }
        }
#endif
    }

    /**
     * @brief Get performance statistics
     */
    struct Stats {
        size_t frames_processed;
        double avg_dct_time_ms;
        double avg_me_time_ms;
        double gpu_utilization;
    };

    Stats get_stats() const {
        return stats_;
    }

private:
    int width_;
    int height_;
    bool initialized_;

#ifdef HAVE_SYCL
    uint8_t* sycl_buffer_ = nullptr;
#endif

    Stats stats_ = {};
};

/**
 * @brief Example: Process video file with SYCL acceleration
 */
void process_video_example(const char* filename) {
    std::cout << "=== FFmpeg + SYCL Integration Example ===" << std::endl;
    std::cout << "Processing: " << filename << std::endl;

#ifdef HAVE_SYCL
    // Create processor for 1920x1080 video
    SYCLFrameProcessor processor(1920, 1080);

    if (!processor.initialize()) {
        std::cerr << "Failed to initialize SYCL processor" << std::endl;
        return;
    }

    // Allocate buffers
    std::vector<uint8_t> frame_data(1920 * 1080 * 3 / 2);
    std::vector<int32_t> dct_output(1920 * 1080);
    std::vector<int> mv_x((1920 / 16) * (1080 / 16));
    std::vector<int> mv_y((1920 / 16) * (1080 / 16));
    std::vector<uint32_t> sad((1920 / 16) * (1080 / 16));

    // Process frame (in real usage, this would be in a loop over video frames)
    processor.process_dct(frame_data.data(), dct_output.data());

    // Motion estimation (need two frames)
    processor.motion_estimation(frame_data.data(), frame_data.data(),
                                mv_x.data(), mv_y.data(), sad.data());

    std::cout << "Frame processed successfully!" << std::endl;
#else
    std::cout << "SYCL support not available in this build" << std::endl;
#endif
}

}  // namespace ffmpeg
}  // namespace avm

int main(int argc, char* argv[]) {
    const char* filename = argc > 1 ? argv[1] : "test.yuv";
    avm::ffmpeg::process_video_example(filename);
    return 0;
}
