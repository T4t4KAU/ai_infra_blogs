#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

#define TOL 1e-5f

// Row-major linear index
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// Vectorized float4 load/store helper
#define FETCH_FLOAT4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])

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
__global__ void sgemm_kernel_v4(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

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
     * Shared double buffer for A and B tiles
     * ------------------------------ */
    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];

    /* ------------------------------
     * Move global pointers to the
     * beginning of the current block tile
     * ------------------------------ */
    float *A_block = A + block_row * BM * K;
    float *B_block = B + block_col * BN;
    float *C_block = C + block_row * BM * N + block_col * BN;

    // per-thread load partition
    const int ldg_a_num = BK * BM / num_threads / 4;
    const int ldg_b_num = BK * BN / num_threads / 4;

    /* ------------------------------
     * Thread mapping for loading A tile
     * ------------------------------ */
    const int a_tile_row = threadIdx.x / (BK / 4);
    const int a_tile_col = (threadIdx.x % (BK / 4)) * 4;
    const int a_tile_stride = BM / ldg_a_num;
    const int b_tile_row = threadIdx.x / (BN / 4);
    const int b_tile_col = (threadIdx.x % (BN / 4)) * 4;
    const int b_tile_stride = BK / ldg_b_num;

    float accum[TM][TN] = {0.0f};

    // register fragments
    float a_frag[2][TM];
    float b_frag[2][TN];

    // staging regs for global vector loads
    float ldg_a_reg[4 * ldg_a_num];
    float ldg_b_reg[4 * ldg_b_num];

#pragma unroll
    for (int i = 0; i < BM; i += a_tile_stride) {
        const int reg_idx = (i / a_tile_stride) * 4;
        FETCH_FLOAT4(ldg_a_reg[reg_idx]) = FETCH_FLOAT4(A_block[OFFSET(a_tile_row + i, a_tile_col, K)]);

        // store A into shared as transposed
        As[0][OFFSET(a_tile_col + 0, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 0];
        As[0][OFFSET(a_tile_col + 1, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 1];
        As[0][OFFSET(a_tile_col + 2, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 2];
        As[0][OFFSET(a_tile_col + 3, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 3];
    }

#pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride) {
        FETCH_FLOAT4(Bs[0][OFFSET(b_tile_row + i, b_tile_col, BN)])
            = FETCH_FLOAT4(B_block[OFFSET(b_tile_row + i, b_tile_col, N)]);
    }

    __syncthreads();

    // preload frag for t=0 from shared buffer 0
#pragma unroll
    for (int m = 0; m < TM; m += 4) { FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[0][OFFSET(0, ty + m, BM)]); }
#pragma unroll
    for (int n = 0; n < TN; n += 4) { FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[0][OFFSET(0, tx + n, BN)]); }

    int write_index = 1; // next shared buffer to write
    int k_base = 0;

    // ---------------------------
    // Main loop over K tiles
    // ---------------------------
    do {
        const int next_k = k_base + BK;

        // prefetch next tile from global into registers
        if (next_k < K) {
#pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride) {
                int reg_idx = (i / a_tile_stride) * 4;

                FETCH_FLOAT4(ldg_a_reg[reg_idx])
                    = FETCH_FLOAT4(A_block[OFFSET(a_tile_row + i, next_k + a_tile_col, K)]);
            }
#pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride) {
                int reg_idx = (i / b_tile_stride) * 4;
                FETCH_FLOAT4(ldg_b_reg[reg_idx])
                    = FETCH_FLOAT4(B_block[OFFSET(next_k + b_tile_row + i, b_tile_col, N)]);
            }
        }

        // shared buffer we are reading (current tile)
        const int load_index = write_index ^ 1;

#pragma unroll
        for (int t = 0; t < BK - 1; ++t) {
            // load next k-frag while computing current frag
#pragma unroll
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[(t + 1) & 1][m]) = FETCH_FLOAT4(As[load_index][OFFSET(t + 1, ty + m, BM)]);
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[(t + 1) & 1][n]) = FETCH_FLOAT4(Bs[load_index][OFFSET(t + 1, tx + n, BN)]);
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) { accum[i][j] += a_frag[t & 1][i] * b_frag[t & 1][j]; }
            }
        }

        if (next_k < K) {
#pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride) {
                int reg_idx = (i / a_tile_stride) * 4;

                As[write_index][OFFSET(a_tile_col + 0, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 0];
                As[write_index][OFFSET(a_tile_col + 1, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 1];
                As[write_index][OFFSET(a_tile_col + 2, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 2];
                As[write_index][OFFSET(a_tile_col + 3, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 3];
            }
#pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride) {
                int reg_idx = (i / b_tile_stride) * 4;

                FETCH_FLOAT4(Bs[write_index][OFFSET(b_tile_row + i, b_tile_col, BN)])
                    = FETCH_FLOAT4(ldg_b_reg[reg_idx]);
            }

            __syncthreads();

            // preload frag for next tile's t=0
#pragma unroll
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[write_index][OFFSET(0, ty + m, BM)]);
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[write_index][OFFSET(0, tx + n, BN)]);
            }

            write_index ^= 1;
        }
#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
            for (int j = 0; j < TN; ++j) { accum[i][j] += a_frag[(BK - 1) & 1][i] * b_frag[(BK - 1) & 1][j]; }
        }

        k_base = next_k;
    } while (k_base < K);

    // Write back C
#pragma unroll
    for (int m = 0; m < TM; ++m) {
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            float4 c4 = FETCH_FLOAT4(C_block[OFFSET(ty + m, tx + n, N)]);
            c4.x = alpha * accum[m][n + 0] + beta * c4.x;
            c4.y = alpha * accum[m][n + 1] + beta * c4.y;
            c4.z = alpha * accum[m][n + 2] + beta * c4.z;
            c4.w = alpha * accum[m][n + 3] + beta * c4.w;
            FETCH_FLOAT4(C_block[OFFSET(ty + m, tx + n, N)]) = c4;
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
        sgemm_kernel_v4<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
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
        sgemm_kernel_v4<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
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

    printf("sgemm_kernel_v4:\n");
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
