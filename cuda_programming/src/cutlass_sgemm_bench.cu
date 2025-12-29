#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

#define CHECK_CUDA_ERROR(call)                                                                                         \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

int main() {
    /* =============================
     * Matrix size
     * ============================= */
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    size_t bytes_A = size_t(M) * K * sizeof(float);
    size_t bytes_B = size_t(K) * N * sizeof(float);
    size_t bytes_C = size_t(M) * N * sizeof(float);

    /* =============================
     * Host allocation & init
     * ============================= */
    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    float *h_C = (float *)malloc(bytes_C);

    for (int i = 0; i < M * K; ++i) { h_A[i] = 1.0f; }
    for (int i = 0; i < K * N; ++i) { h_B[i] = 2.0f; }
    for (int i = 0; i < M * N; ++i) { h_C[i] = 0.0f; }

    /* =============================
     * Device allocation
     * ============================= */
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, bytes_C));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

    /* =============================
     * CUTLASS GEMM 定义（SIMT）
     * ============================= */
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, cutlass::arch::OpClassSimt,
        cutlass::arch::Sm120, cutlass::gemm::GemmShape<128, 128, 8>, // Threadblock tile
        cutlass::gemm::GemmShape<64, 64, 8>,                         // Warp tile
        cutlass::gemm::GemmShape<1, 1, 1>,                           // Instruction tile (SIMT)
        cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

    Gemm gemm_op;

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename Gemm::Arguments args(problem_size, {d_A, K}, {d_B, N}, {d_C, N}, {d_C, N}, {alpha, beta});

    /* =============================
     * Warmup
     * ============================= */
    const int warmup_iters = 10;
    for (int i = 0; i < warmup_iters; ++i) {
        cutlass::Status status = gemm_op(args);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS GEMM failed\n";
            return -1;
        }
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    /* =============================
     * Timed run
     * ============================= */
    const int repeat_iters = 10;

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < repeat_iters; ++i) {
        cutlass::Status status = gemm_op(args);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS GEMM failed\n";
            return -1;
        }
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start, stop));

    float avg_ms = elapsed_ms / repeat_iters;

    /* =============================
     * GFLOPS
     * ============================= */
    double flops = 2.0 * double(M) * N * K;
    double gflops = flops / (avg_ms * 1e6);

    printf("CUTLASS SGEMM (SIMT FP32):\n");
    printf("  Avg time: %.3f ms\n", avg_ms);
    printf("  Perf:     %.2f GFLOPS\n", gflops);

    /* =============================
     * Cleanup
     * ============================= */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();
    return 0;
}
