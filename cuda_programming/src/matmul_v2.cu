#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call)                                                                                         \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// Block-tiled + register-tiled SGEMM kernel
// Computes: C = alpha * A * B + beta * C
//
// BM, BN : block-level tile size of C (BM rows × BN cols)
// BK     : block-level tile size along K dimension
// TM, TN : per-thread register tile size (TM rows × TN cols)
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_kernel_v2(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

    /* ------------------------------
     * Block coordinates
     * Each block computes one BM×BN tile of C
     * ------------------------------ */
    const int block_col = blockIdx.x; // tile index in N dimension
    const int block_row = blockIdx.y; // tile index in M dimension

    /* ------------------------------
     * Thread layout inside a block
     * Threads are logically arranged in 2D:
     *   (BM / TM) × (BN / TN)
     * ------------------------------ */
    const int threads_per_row = BN / TN;
    const int threads_per_col = BM / TM;
    const int num_threads = threads_per_row * threads_per_col;

    /* ------------------------------
     * Per-thread output tile offset
     * Each thread computes a TM×TN sub-tile of C
     * ------------------------------ */
    const int tx = (threadIdx.x % threads_per_row) * TN; // column offset
    const int ty = (threadIdx.x / threads_per_row) * TM; // row offset

    /* ------------------------------
     * Shared memory for A and B tiles
     * As : BM × BK
     * Bs : BK × BN
     * ------------------------------ */
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    /* ------------------------------
     * Move global pointers to the
     * beginning of the current block tile
     * ------------------------------ */
    float *A_block = A + block_row * BM * K;
    float *B_block = B + block_col * BN;
    float *C_block = C + block_row * BM * N + block_col * BN;

    /* ------------------------------
     * Thread mapping for loading A tile
     * ------------------------------ */
    const int a_tile_row = threadIdx.x / BK;
    const int a_tile_col = threadIdx.x % BK;
    const int a_tile_stride = num_threads / BK;
    const int b_tile_row = threadIdx.x / BN;
    const int b_tile_col = threadIdx.x % BN;
    const int b_tile_stride = num_threads / BN;

    float accum[TM][TN] = {0.0f};

#pragma unroll
    for (int k = 0; k < K; k += BK) {
        /* ------------------------------
         * Load A tile into shared memory
         * ------------------------------ */
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A_block[(a_tile_row + i) * K + a_tile_col];
        }
        /* ------------------------------
         * Load B tile into shared memory
         * ------------------------------ */
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B_block[(b_tile_row + i) * N + b_tile_col];
        }

        __syncthreads(); // Ensure As and Bs are fully loaded

        /* ------------------------------
         * Advance A and B pointers
         * ------------------------------ */
        A_block += BK;
        B_block += BK * N;

        /* ------------------------------
         * Compute: register-level GEMM
         * ------------------------------ */
#pragma unroll
        for (int t = 0; t < BK; ++t) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) { accum[i][j] += As[(ty + i) * BK + t] * Bs[t * BN + (tx + j)]; }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            // C_block(ty + i, tx + j) =
            // alpha × accum(i, j) + beta × C_block(ty + i, tx + j)
            C_block[(ty + i) * N + (tx + j)] = alpha * accum[i][j] + beta * C_block[(ty + i) * N + (tx + j)];
        }
    }
}

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

    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    /* =============================
     * Host allocation
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
     * Kernel launch config
     * ============================= */
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(256);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    /* =============================
     * Warmup (排除冷启动)
     * ============================= */
    const int warmup_iters = 10;
    for (int i = 0; i < warmup_iters; ++i) {
        sgemm_kernel_v2<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    /* =============================
     * Timed run (steady-state)
     * ============================= */
    const int repeat_iters = 10;

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < repeat_iters; ++i) {
        sgemm_kernel_v2<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start, stop));

    float avg_ms = elapsed_ms / repeat_iters;

    /* =============================
     * GFLOPS
     * ============================= */
    double flops = 2.0 * M * N * K;
    double gflops = flops / (avg_ms * 1e6);

    printf("sgemm_kernel_v2:\n");
    printf("  Avg time: %.3f ms\n", avg_ms);
    printf("  Perf:     %.2f GFLOPS\n", gflops);

    /* =============================
     * Cleanup
     * ============================= */
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
