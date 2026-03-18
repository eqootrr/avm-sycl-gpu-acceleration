// examples/integration/opencv_integration.cpp
// OpenCV integration example for SYCL GPU acceleration

/*
 * This example demonstrates how to integrate SYCL GPU acceleration
 * with OpenCV for real-time video processing.
 *
 * Prerequisites:
 * - OpenCV 4.x (libopencv-dev)
 * - SYCL runtime (Intel DPC++ or AdaptiveCpp)
 *
 * Build:
 *   g++ -std=c++17 opencv_integration.cpp `pkg-config --cflags --libs opencv4` -o opencv_integration
 *   # Or with CMake:
 *   cmake .. -DAVM_BUILD_EXAMPLES=ON
 */

#include <iostream>
#include <vector>
#include <chrono>

// SYCL headers
#ifdef HAVE_SYCL
#include "sycl_context.hpp"
#include "sycl_txfm.hpp"
#include "sycl_me.hpp"
#include "sycl_wrapper.hpp"
#endif

// OpenCV headers (commented out for standalone compilation)
// #include <opencv2/opencv.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>

namespace avm {
namespace opencv {

/**
 * @brief SYCL-accelerated image processor for OpenCV
 *
 * Provides GPU-accelerated operations that can be used with cv::Mat
 */
class SYCLImageProcessor {
public:
    SYCLImageProcessor() : initialized_(false) {}
    ~SYCLImageProcessor() = default;

    /**
     * @brief Initialize SYCL context
     */
    bool initialize() {
#ifdef HAVE_SYCL
        if (!avm::sycl::should_use_sycl()) {
            return false;
        }

        auto& ctx = avm::sycl::SYCLContext::instance();
        if (!ctx.initialize()) {
            return false;
        }

        std::cout << "SYCL GPU: " << ctx.backend_name() << std::endl;
        initialized_ = true;
        return true;
#else
        return false;
#endif
    }

    /**
     * @brief Compute DCT of image blocks (useful for compression)
     * @param gray_image Grayscale image data (8-bit)
     * @param width Image width
     * @param height Image height
     * @param dct_coefficients Output DCT coefficients
     */
    void compute_block_dct(const uint8_t* gray_image, int width, int height,
                           std::vector<int32_t>& dct_coefficients) {
#ifdef HAVE_SYCL
        if (!initialized_) return;

        auto& ctx = avm::sycl::SYCLContext::instance();
        auto& q = ctx.queue();

        int blocks_x = width / 8;
        int blocks_y = height / 8;
        dct_coefficients.resize(blocks_x * blocks_y * 64);

        auto start = std::chrono::high_resolution_clock::now();

        for (int by = 0; by < blocks_y; ++by) {
            for (int bx = 0; bx < blocks_x; ++bx) {
                std::vector<int16_t> block(64);
                for (int y = 0; y < 8; ++y) {
                    for (int x = 0; x < 8; ++x) {
                        int idx = (by * 8 + y) * width + (bx * 8 + x);
                        block[y * 8 + x] = static_cast<int16_t>(gray_image[idx]) - 128;
                    }
                }
                avm::sycl::fdct8x8(q, block.data(), &dct_coefficients[(by * blocks_x + bx) * 64]);
            }
        }
        q.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "DCT time: " << duration.count() << " μs" << std::endl;
#endif
    }

    /**
     * @brief Compute motion vectors between two frames
     * @param prev_frame Previous grayscale frame
     * @param curr_frame Current grayscale frame
     * @param width Frame width
     * @param height Frame height
     * @param motion_vectors Output motion vectors (dx, dy pairs)
     */
    void compute_motion_vectors(const uint8_t* prev_frame, const uint8_t* curr_frame,
                                int width, int height,
                                std::vector<std::pair<int16_t, int16_t>>& motion_vectors) {
#ifdef HAVE_SYCL
        if (!initialized_) return;

        auto& ctx = avm::sycl::SYCLContext::instance();
        auto& q = ctx.queue();

        int blocks_x = width / 16;
        int blocks_y = height / 16;
        motion_vectors.resize(blocks_x * blocks_y);

        auto start = std::chrono::high_resolution_clock::now();

        for (int by = 0; by < blocks_y; ++by) {
            for (int bx = 0; bx < blocks_x; ++bx) {
                // Full search in 32x32 window
                uint32_t best_sad = UINT32_MAX;
                int best_dx = 0, best_dy = 0;

                for (int dy = -16; dy <= 16; ++dy) {
                    for (int dx = -16; dx <= 16; ++dx) {
                        int ref_x = bx * 16 + dx;
                        int ref_y = by * 16 + dy;

                        if (ref_x < 0 || ref_x + 16 > width) continue;
                        if (ref_y < 0 || ref_y + 16 > height) continue;

                        std::vector<uint8_t> ref_block(256), cur_block(256);
                        for (int y = 0; y < 16; ++y) {
                            for (int x = 0; x < 16; ++x) {
                                ref_block[y * 16 + x] = prev_frame[(ref_y + y) * width + (ref_x + x)];
                                cur_block[y * 16 + x] = curr_frame[(by * 16 + y) * width + (bx * 16 + x)];
                            }
                        }

                        uint32_t sad = avm::sycl::sad16x16(q, ref_block.data(), cur_block.data());

                        if (sad < best_sad) {
                            best_sad = sad;
                            best_dx = dx;
                            best_dy = dy;
                        }
                    }
                }

                motion_vectors[by * blocks_x + bx] = {static_cast<int16_t>(best_dx),
                                                       static_cast<int16_t>(best_dy)};
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Motion estimation time: " << duration.count() << " ms" << std::endl;
#endif
    }

    /**
     * @brief Visualize motion vectors (for debugging/display)
     * @param frame Input frame (will be modified)
     * @param width Frame width
     * @param height Frame height
     * @param motion_vectors Motion vectors to visualize
     */
    void visualize_motion_vectors(uint8_t* frame, int width, int height,
                                  const std::vector<std::pair<int16_t, int16_t>>& motion_vectors) {
        int blocks_x = width / 16;
        int blocks_y = height / 16;

        for (int by = 0; by < blocks_y; ++by) {
            for (int bx = 0; bx < blocks_x; ++bx) {
                const auto& mv = motion_vectors[by * blocks_x + bx];

                // Draw arrow from block center
                int cx = bx * 16 + 8;
                int cy = by * 16 + 8;
                int ex = cx + mv.first;
                int ey = cy + mv.second;

                // Simple line drawing (Bresenham's algorithm simplified)
                // In real OpenCV: cv::arrowedLine()
                if (mv.first != 0 || mv.second != 0) {
                    // Mark block as having motion
                    for (int y = by * 16; y < by * 16 + 16; ++y) {
                        for (int x = bx * 16; x < bx * 16 + 16; ++x) {
                            // Draw border
                            if (y == by * 16 || y == by * 16 + 15 ||
                                x == bx * 16 || x == bx * 16 + 15) {
                                frame[y * width + x] = 255;  // White border
                            }
                        }
                    }
                }
            }
        }
    }

private:
    bool initialized_;
};

/**
 * @brief Example: Real-time motion detection with OpenCV + SYCL
 */
void motion_detection_example() {
    std::cout << "=== OpenCV + SYCL Motion Detection Example ===" << std::endl;

#ifdef HAVE_SYCL
    SYCLImageProcessor processor;

    if (!processor.initialize()) {
        std::cerr << "SYCL initialization failed" << std::endl;
        return;
    }

    // Simulate video frames (in real usage, use cv::VideoCapture)
    int width = 640;
    int height = 480;
    std::vector<uint8_t> prev_frame(width * height, 128);
    std::vector<uint8_t> curr_frame(width * height, 128);

    // Simulate motion in center
    for (int y = 200; y < 280; ++y) {
        for (int x = 280; x < 360; ++x) {
            curr_frame[y * width + x] = 200;  // Brighter region
        }
    }

    // Compute motion vectors
    std::vector<std::pair<int16_t, int16_t>> motion_vectors;
    processor.compute_motion_vectors(prev_frame.data(), curr_frame.data(),
                                     width, height, motion_vectors);

    // Visualize
    processor.visualize_motion_vectors(curr_frame.data(), width, height, motion_vectors);

    // Count blocks with significant motion
    int motion_blocks = 0;
    for (const auto& mv : motion_vectors) {
        if (std::abs(mv.first) > 2 || std::abs(mv.second) > 2) {
            motion_blocks++;
        }
    }

    std::cout << "Blocks with motion: " << motion_blocks << " / " << motion_vectors.size() << std::endl;

    // In real OpenCV:
    // cv::Mat frame(height, width, CV_8UC1, curr_frame.data());
    // cv::imshow("Motion Detection", frame);
    // cv::waitKey(0);

#else
    std::cout << "SYCL support not available in this build" << std::endl;
#endif
}

/**
 * @brief Example: Image compression with DCT
 */
void image_compression_example() {
    std::cout << "=== OpenCV + SYCL Image Compression Example ===" << std::endl;

#ifdef HAVE_SYCL
    SYCLImageProcessor processor;

    if (!processor.initialize()) {
        std::cerr << "SYCL initialization failed" << std::endl;
        return;
    }

    // Create test image (gradient)
    int width = 512;
    int height = 512;
    std::vector<uint8_t> image(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image[y * width + x] = (x + y) / 4;
        }
    }

    // Compute DCT
    std::vector<int32_t> dct_coeffs;
    processor.compute_block_dct(image.data(), width, height, dct_coeffs);

    // Simple quantization (keep only low frequencies)
    int nonzero = 0;
    for (auto& coeff : dct_coeffs) {
        if (std::abs(coeff) < 10) {
            coeff = 0;  // Zero out small coefficients
        } else {
            nonzero++;
        }
    }

    double compression_ratio = 100.0 * (1.0 - static_cast<double>(nonzero) / dct_coeffs.size());
    std::cout << "Compression: " << compression_ratio << "% coefficients zeroed" << std::endl;

#else
    std::cout << "SYCL support not available in this build" << std::endl;
#endif
}

}  // namespace opencv
}  // namespace avm

int main() {
    avm::opencv::motion_detection_example();
    std::cout << std::endl;
    avm::opencv::image_compression_example();
    return 0;
}
