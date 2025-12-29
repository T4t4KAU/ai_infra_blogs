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

constexpr int WARP_SIZE = 32;

// Block-tiled + Warp-tiled + Reg-tiled SGEMM
// C = alpha*A*B + beta*C
//
// BM, BN : block tile (BM x BN)
// BK     : K tile
// WM, WN : warp tile (WM x WN)  (each warp computes one WMxWN sub-tile of the block)
// WNITER : warp splits WN into WNITER groups, each group size WSUBN = WN/WNITER
// TM, TN : per-thread reg tile (TM x TN), TN must be multiple of 4
// NUM_THREADS : block threads, should be (BM/WM)*(BN/WN)*32
template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WNITER, const int TM,
          const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemm_kernel_v5(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int M, int N,
                    int K, float alpha, float beta) {
    // block tile coords on C
    const int block_col = blockIdx.x; // along N
    const int block_row = blockIdx.y; // along M

    // warps in this block
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    constexpr int WARPS_PER_COL = BN / WN;
    // constexpr int WARPS_PER_ROW = BM / WM;

    // (warp_row, warp_col) within block tile
    const int warp_col = warp_idx % WARPS_PER_COL;
    const int warp_row = warp_idx / WARPS_PER_COL;

    // Warp micro-tiling
    // WMITER is derived: how many "sub-rows" we iterate in WM direction per warp step
    // WSUBM/WSUBN define warp's internal subdivision
    constexpr int WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    // lane -> (thread_row_in_warp, thread_col_in_warp)
    // warp covers WSUBM x WSUBN per (w_sub_row_idx, w_sub_col_idx) region,
    // each thread computes TMxTN
    constexpr int THREADS_PER_WSUBN = WSUBN / TN;
    const int thread_col_in_warp = lane % THREADS_PER_WSUBN;
    const int thread_row_in_warp = lane / THREADS_PER_WSUBN;

    // shared double buffer: As is stored transposed (k-major)
    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];

    // move base pointers to this block tile
    const float *A_block = A + (block_row * BM) * K;
    const float *B_block = B + (block_col * BN);
    float *C_block = C + (block_row * BM) * N + (block_col * BN);

    // ---------- cooperative global->shared load mapping ----------
    // Each thread loads float4s to cover As and Bs.
    // A tile: BM x BK, but stored in shared as (BK x BM) for coalesced K access
    constexpr int a_vec_per_tile = (BM * BK) / 4;
    constexpr int b_vec_per_tile = (BK * BN) / 4;

    constexpr int a_vec_per_thread = a_vec_per_tile / NUM_THREADS;
    constexpr int b_vec_per_thread = b_vec_per_tile / NUM_THREADS;

    // per-thread linear vector indices
    // For A: vector index corresponds to (m, k4) where k4 steps by 4
    // For B: vector index corresponds to (k, n4)
    float a_ldg_reg[4 * a_vec_per_thread];
    float b_ldg_reg[4 * b_vec_per_thread];

    // per-thread accumulators
    // warp computes WMxWN; each thread owns WMITER*TM rows and WNITER*TN cols
    float thread_results[WMITER * TM * WNITER * TN];
#pragma unroll
    for (int i = 0; i < (WMITER * TM * WNITER * TN); ++i) { thread_results[i] = 0.f; }

    // register fragments
    float reg_m[WMITER * TM];
    float reg_n[WNITER * TN];

    int write_buf = 0;

    // pre-load first tile into shared
    {
        // A: load float4s from global, store into As transposed: As[k*BM + m]
#pragma unroll
        for (int v = 0; v < a_vec_per_thread; ++v) {
            int vec_idx = threadIdx.x + v * NUM_THREADS; // [0, a_vec_per_tile)
            int m = vec_idx / (BK / 4);
            int k4 = vec_idx % (BK / 4); // 4-float pack along K

            // global A index: (m, k4*4)
            float4 a4 = reinterpret_cast<const float4 *>(&A_block[OFFSET(m, k4 * 4, K)])[0];

            // shared store transposed: for each element at k, put to As[k*BM + m]
            As[write_buf][OFFSET(k4 * 4 + 0, m, BM)] = a4.x;
            As[write_buf][OFFSET(k4 * 4 + 1, m, BM)] = a4.y;
            As[write_buf][OFFSET(k4 * 4 + 2, m, BM)] = a4.z;
            As[write_buf][OFFSET(k4 * 4 + 3, m, BM)] = a4.w;
        }

        // B: load float4s from global, store into Bs as [k*BN + n]
#pragma unroll
        for (int v = 0; v < b_vec_per_thread; ++v) {
            int vec_idx = threadIdx.x + v * NUM_THREADS; // [0, b_vec_per_tile)
            int k = vec_idx / (BN / 4);
            int n4 = vec_idx % (BN / 4);

            reinterpret_cast<float4 *>(&Bs[write_buf][OFFSET(k, n4 * 4, BN)])[0]
                = reinterpret_cast<const float4 *>(&B_block[OFFSET(k, n4 * 4, N)])[0];
        }
    }
    __syncthreads();

    // ---------- main loop over K tiles ----------
    for (int k_base = 0; k_base < K; k_base += BK) {

        // prefetch next tile into regs (global -> regs), then commit to other shared buffer
        const int next_k = k_base + BK;
        const int next_buf = write_buf ^ 1;

        if (next_k < K) {
            const float *A_next = A_block + next_k;
            const float *B_next = B_block + next_k * N;

#pragma unroll
            for (int v = 0; v < a_vec_per_thread; ++v) {
                int vec_idx = threadIdx.x + v * NUM_THREADS;
                int m = vec_idx / (BK / 4);
                int k4 = vec_idx % (BK / 4);
                FETCH_FLOAT4(a_ldg_reg[v * 4]) = reinterpret_cast<const float4 *>(&A_next[OFFSET(m, k4 * 4, K)])[0];
            }

#pragma unroll
            for (int v = 0; v < b_vec_per_thread; ++v) {
                int vec_idx = threadIdx.x + v * NUM_THREADS;
                int k = vec_idx / (BN / 4);
                int n4 = vec_idx % (BN / 4);
                FETCH_FLOAT4(b_ldg_reg[v * 4]) = reinterpret_cast<const float4 *>(&B_next[OFFSET(k, n4 * 4, N)])[0];
            }
        }

        // compute using current shared buffer = write_buf
        const int read_buf = write_buf;

#pragma unroll
        for (int dot = 0; dot < BK; ++dot) {

            // load A regs: As[dot, m] where m is warp-local WM rows
#pragma unroll
            for (int wsr = 0; wsr < WMITER; ++wsr) {
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const int m_in_block = warp_row * WM + wsr * WSUBM + thread_row_in_warp * TM + i;
                    reg_m[wsr * TM + i] = As[read_buf][OFFSET(dot, m_in_block, BM)];
                }
            }

            // load B regs: Bs[dot, n] where n is warp-local WN cols
#pragma unroll
            for (int wsc = 0; wsc < WNITER; ++wsc) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int n_in_block = warp_col * WN + wsc * WSUBN + thread_col_in_warp * TN + j;
                    reg_n[wsc * TN + j] = Bs[read_buf][OFFSET(dot, n_in_block, BN)];
                }
            }

            // FMA into thread_results
#pragma unroll
            for (int wsr = 0; wsr < WMITER; ++wsr) {
#pragma unroll
                for (int wsc = 0; wsc < WNITER; ++wsc) {
#pragma unroll
                    for (int i = 0; i < TM; ++i) {
#pragma unroll
                        for (int j = 0; j < TN; ++j) {
                            thread_results[(wsr * TM + i) * (WNITER * TN) + (wsc * TN + j)]
                                += reg_m[wsr * TM + i] * reg_n[wsc * TN + j];
                        }
                    }
                }
            }
        }

        // commit prefetched regs to the other shared buffer
        if (next_k < K) {
#pragma unroll
            for (int v = 0; v < a_vec_per_thread; ++v) {
                int vec_idx = threadIdx.x + v * NUM_THREADS;
                int m = vec_idx / (BK / 4);
                int k4 = vec_idx % (BK / 4);

                float4 a4 = FETCH_FLOAT4(a_ldg_reg[v * 4]);
                As[next_buf][OFFSET(k4 * 4 + 0, m, BM)] = a4.x;
                As[next_buf][OFFSET(k4 * 4 + 1, m, BM)] = a4.y;
                As[next_buf][OFFSET(k4 * 4 + 2, m, BM)] = a4.z;
                As[next_buf][OFFSET(k4 * 4 + 3, m, BM)] = a4.w;
            }

#pragma unroll
            for (int v = 0; v < b_vec_per_thread; ++v) {
                int vec_idx = threadIdx.x + v * NUM_THREADS;
                int k = vec_idx / (BN / 4);
                int n4 = vec_idx % (BN / 4);
                reinterpret_cast<float4 *>(&Bs[next_buf][OFFSET(k, n4 * 4, BN)])[0] = FETCH_FLOAT4(b_ldg_reg[v * 4]);
            }

            __syncthreads();
            write_buf ^= 1;
        }
    }

    // ---------- write back (alpha/beta fused), vectorized ----------
    // C origin for this warp
    float *C_warp = C_block + (warp_row * WM) * N + (warp_col * WN);

#pragma unroll
    for (int wsr = 0; wsr < WMITER; ++wsr) {
#pragma unroll
        for (int wsc = 0; wsc < WNITER; ++wsc) {

            float *C_sub = C_warp + (wsr * WSUBM) * N + (wsc * WSUBN);

#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; j += 4) {

                    const int c_row = thread_row_in_warp * TM + i;
                    const int c_col = thread_col_in_warp * TN + j;

                    float4 c4 = reinterpret_cast<float4 *>(&C_sub[OFFSET(c_row, c_col, N)])[0];

                    const int base = (wsr * TM + i) * (WNITER * TN) + (wsc * TN + j);

                    c4.x = alpha * thread_results[base + 0] + beta * c4.x;
                    c4.y = alpha * thread_results[base + 1] + beta * c4.y;
                    c4.z = alpha * thread_results[base + 2] + beta * c4.z;
                    c4.w = alpha * thread_results[base + 3] + beta * c4.w;

                    reinterpret_cast<float4 *>(&C_sub[OFFSET(c_row, c_col, N)])[0] = c4;
                }
            }
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
    constexpr int TN = 4;
    constexpr int WM = 64;
    constexpr int WN = 64;
    constexpr int WNITER = 4;
    constexpr int NUM_THREADS = 128;

    dim3 blockDim(NUM_THREADS);
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    /* =============================
     * Warmup (排除冷启动)
     * ============================= */
    const int warmup_iters = 10;
    for (int i = 0; i < warmup_iters; ++i) {
        sgemm_kernel_v5<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
            <<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
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
        sgemm_kernel_v5<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
            <<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
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
