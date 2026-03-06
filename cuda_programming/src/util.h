#pragma once
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

// ------------------------------------------------------------
// Minimal CUDA error checking for helper utilities
// ------------------------------------------------------------
#ifndef CUDA_UTIL_CHECK
#define CUDA_UTIL_CHECK(call)                                                                                          \
    do {                                                                                                               \
        cudaError_t _err = (call);                                                                                     \
        if (_err != cudaSuccess) {                                                                                     \
            std::printf("CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(_err));                  \
        }                                                                                                              \
    } while (0)
#endif

// ------------------------------------------------------------
// all_close: elementwise comparison with tolerance
// Returns true if |A - B| <= atol for all elements
// ------------------------------------------------------------
inline bool all_close(const float *A, const float *B, int m, int n, float atol = 1e-3f) {
    const size_t total = static_cast<size_t>(m) * static_cast<size_t>(n);

    for (size_t idx = 0; idx < total; ++idx) {
        const float diff = std::fabs(A[idx] - B[idx]);
        if (diff > atol) {
            const int row = static_cast<int>(idx / n);
            const int col = static_cast<int>(idx % n);
            std::printf("Mismatch at (%d, %d) [flat=%zu]: A=%f, B=%f, |diff|=%f\n", row, col, idx, A[idx], B[idx],
                        diff);
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------
// print_host_matrix: print a row-major matrix on host
// ------------------------------------------------------------
inline void print_host_matrix(const float *matrix, int m, int n, const char *name = nullptr, int precision = 6) {
    if (name) {
        std::printf("%s [%d x %d]\n", name, m, n);
    }

    const char *fmt = (precision <= 6) ? "%.6f " : "%.9f ";

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) { std::printf(fmt, matrix[i * n + j]); }
        std::printf("\n");
    }
}

// ------------------------------------------------------------
// print_device_matrix: copy device matrix to host and print
// Note: This is for debugging only; copying from device is slow.
// ------------------------------------------------------------
inline void print_device_matrix(const float *dev_ptr, int m, int n, const char *name = nullptr, int precision = 6) {
    const size_t total = static_cast<size_t>(m) * static_cast<size_t>(n);

    std::vector<float> host(total);

    CUDA_UTIL_CHECK(cudaMemcpy(host.data(), dev_ptr, total * sizeof(float), cudaMemcpyDeviceToHost));

    // If memcpy failed, printing host will be meaningless; we still print,
    // but the CUDA_UTIL_CHECK above will report the error.
    print_host_matrix(host.data(), m, n, name, precision);
}