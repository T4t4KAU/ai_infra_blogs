#include <cuda_runtime.h>
#include <iostream>

const int BLOCK_SIZE = 16;

#define CHECK_CUDA_ERROR(call)                                                                                         \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

__global__ void sgemm_kernel_v1(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum_val = 0.f;
        for (int k = 0; k < K; ++k) { sum_val += A[row * K + k] * B[k * N + col]; }
        C[row * N + col] = alpha * sum_val + beta * C[row * N + col];
    }
}

int main() {
    const int M = 4096;
    const int K = 4096;
    const int N = 4096;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    float *h_C = (float *)malloc(bytes_C);

    for (int i = 0; i < M * K; ++i) { h_A[i] = 1.0f; }
    for (int i = 0; i < K * N; ++i) { h_B[i] = 2.0f; }
    for (int i = 0; i < M * N; ++i) { h_C[i] = 0.0f; }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, bytes_C));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    /* =====================================================
     * 1. Warmup（不计时）
     * ===================================================== */
    const int warmup_iters = 10;
    for (int i = 0; i < warmup_iters; ++i) { sgemm_kernel_v1<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta); }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    /* =====================================================
     * 2. 正式计时（稳定态）
     * ===================================================== */
    const int repeat_iters = 10;

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < repeat_iters; ++i) { sgemm_kernel_v1<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta); }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float elapsed_ms = 0.f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start, stop));

    float avg_ms = elapsed_ms / repeat_iters;

    /* =====================================================
     * 3. 计算 GFLOPS
     * ===================================================== */
    double flops = 2.0 * M * N * K;
    double gflops = flops / (avg_ms * 1e6);

    std::cout << "Average kernel time: " << avg_ms << " ms\n";
    std::cout << "Performance: " << gflops << " GFLOPS\n";

    /* =====================================================
     * Cleanup
     * ===================================================== */
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_A);
    free(h_B);
    free(h_C);

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaDeviceReset());

    return 0;
}
