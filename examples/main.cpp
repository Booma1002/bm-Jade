#include "header/Engine.hpp"
#include <chrono>

using namespace bm;

int main() {
    LOG_INFO("[Thermal Throttler] Initiating Core Meltdown Sequence...");
    Jade::set_seed(42);

    auto start_time = std::chrono::high_resolution_clock::now();
    double running_sum = 0.0;

    // 50 Epochs of absolute brutality
    for (int epoch = 1; epoch <= 50; ++epoch) {
        std::cout << "[Epoch " << epoch << "/50] Igniting 8-Billion Op GEMM..." << std::flush;
        auto ep_start = std::chrono::high_resolution_clock::now();

        // 1. Allocate 3 massive dense blocks (~100MB of pure double-precision data)
        // This tests your Allocator's ability to cleanly map and unmap large contiguous pages.
        Jade W = Jade::randn(DType::FLOAT64, 2048, 2048);
        Jade X = Jade::randn(DType::FLOAT64, 2048, 2048);
        Jade b = Jade::randn(DType::FLOAT64, 2048);

        // 2. The $O(N^3)$ GEMM Grinder. 2048^3 = 8.5 Billion operations per loop.
        Jade Z = W.dot(X);

        // 3. Strided Broadcasting Stress
        // Broadcast a (2048) vector across a (2048, 2048) matrix.
        Z += b;

        // 4. Memory Bending & Map Operations
        // Force the threads to read the massive matrix, compute heavy transcendental math,
        // and write it back out.
        Jade activations = Jade::std(Z); // Arbitrary heavy reduction
        Jade clipped = Jade::max(activations);

        running_sum += clipped.item<double>();

        auto ep_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> ep_diff = ep_end - ep_start;
        std::cout << " Completed in " << ep_diff.count() << "s. Checksum: " << running_sum << "\n";

        // Scope ends here. W, X, b, Z, activations, and clipped all drop their shared_ptrs.
        // If your destructor or allocator is flawed, this loops leaks ~150MB per tick and will OOM kill your machine in seconds.
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "\n[SURVIVED] Total Execution Time: " << diff.count() << " seconds.\n";
    std::cout << "If your PC didn't shut down, your engine is bulletproof.\n";
}