// examples/basic_usage.cpp
// Basic SYCL GPU acceleration example for AV2 codec

#include <iostream>
#include <vector>
#include <random>
#include "../src/sycl_context.hpp"
#include "../src/sycl_txfm.hpp"
#include "../src/sycl_me.hpp"
#include "../src/sycl_wrapper.hpp"

// Generate random test data
void generate_random_block(int16_t* block, size_t size, int16_t min_val = -512, int16_t max_val = 511) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int16_t> dist(min_val, max_val);

    for (size_t i = 0; i < size; ++i) {
        block[i] = dist(gen);
    }
}

int main() {
    std::cout << "=== AV2 SYCL GPU Acceleration Demo ===" << std::endl;
    std::cout << std::endl;

#ifdef HAVE_SYCL
    // Check if SYCL is available
    if (!avm::sycl::should_use_sycl()) {
        std::cout << "SYCL GPU acceleration not available." << std::endl;
        std::cout << "Falling back to CPU path." << std::endl;
        return 0;
    }

    // Initialize SYCL context
    auto& ctx = avm::sycl::SYCLContext::instance();

    std::cout << "Initializing SYCL context..." << std::endl;
    if (!ctx.initialize()) {
        std::cerr << "Failed to initialize SYCL context!" << std::endl;
        return 1;
    }

    // Print device information
    std::cout << std::endl;
    std::cout << "=== Device Information ===" << std::endl;
    std::cout << "Backend: " << ctx.backend_name() << std::endl;
    std::cout << "GPU: " << (ctx.is_gpu() ? "Yes" : "No") << std::endl;
    std::cout << "Compute Units: " << ctx.compute_units() << std::endl;
    std::cout << "Global Memory: " << (ctx.global_mem_size() / 1024 / 1024) << " MB" << std::endl;
    std::cout << std::endl;

    // List all available devices
    std::cout << "=== Available Devices ===" << std::endl;
    auto devices = ctx.list_devices();
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto& dev = devices[i];
        std::cout << i << ": " << dev.name << std::endl;
        std::cout << "   Vendor: " << dev.vendor << std::endl;
        std::cout << "   Type: " << (dev.is_gpu ? "GPU" : "CPU") << std::endl;
        std::cout << "   Compute Units: " << dev.compute_units << std::endl;
    }
    std::cout << std::endl;

    // Demo 1: DCT Transform
    std::cout << "=== Demo 1: DCT 8x8 Transform ===" << std::endl;
    {
        std::vector<int16_t> input(64);
        std::vector<int32_t> output(64);

        generate_random_block(input.data(), 64);
        std::cout << "Input sample values: ";
        for (int i = 0; i < 4; ++i) std::cout << input[i] << " ";
        std::cout << "..." << std::endl;

        // Perform DCT
        avm::sycl::fdct8x8(ctx.queue(), input.data(), output.data());

        std::cout << "DCT output sample values: ";
        for (int i = 0; i < 4; ++i) std::cout << output[i] << " ";
        std::cout << "..." << std::endl;
        std::cout << "DCT transform completed successfully!" << std::endl;
    }
    std::cout << std::endl;

    // Demo 2: Motion Estimation SAD
    std::cout << "=== Demo 2: SAD 16x16 Motion Estimation ===" << std::endl;
    {
        std::vector<uint8_t> ref_block(256);
        std::vector<uint8_t> candidate_block(256);

        generate_random_block(reinterpret_cast<int16_t*>(ref_block.data()), 128, 0, 255);
        generate_random_block(reinterpret_cast<int16_t*>(candidate_block.data()), 128, 0, 255);

        // Compute SAD
        uint32_t sad = avm::sycl::sad16x16(ctx.queue(),
                                           ref_block.data(),
                                           candidate_block.data());

        std::cout << "Reference block sample: ";
        for (int i = 0; i < 4; ++i) std::cout << (int)ref_block[i] << " ";
        std::cout << "..." << std::endl;

        std::cout << "Candidate block sample: ";
        for (int i = 0; i < 4; ++i) std::cout << (int)candidate_block[i] << " ";
        std::cout << "..." << std::endl;

        std::cout << "SAD (Sum of Absolute Differences): " << sad << std::endl;
    }
    std::cout << std::endl;

    // Demo 3: IDCT (Inverse DCT)
    std::cout << "=== Demo 3: IDCT 8x8 Inverse Transform ===" << std::endl;
    {
        std::vector<int32_t> coeffs(64);
        std::vector<int16_t> output(64);

        // Generate sparse coefficients (typical for video)
        for (auto& c : coeffs) c = 0;
        coeffs[0] = 1024;  // DC coefficient
        coeffs[1] = 256;   // First AC coefficient
        coeffs[8] = -128;  // Second AC coefficient

        // Perform IDCT
        avm::sycl::idct8x8(ctx.queue(), coeffs.data(), output.data());

        std::cout << "IDCT output sample values: ";
        for (int i = 0; i < 4; ++i) std::cout << output[i] << " ";
        std::cout << "..." << std::endl;
        std::cout << "IDCT transform completed successfully!" << std::endl;
    }
    std::cout << std::endl;

    std::cout << "=== All demos completed successfully! ===" << std::endl;

#else
    std::cout << "SYCL support not compiled in." << std::endl;
    std::cout << "Please rebuild with -DAVM_ENABLE_SYCL=ON" << std::endl;
#endif

    return 0;
}
