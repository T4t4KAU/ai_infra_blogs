#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) { val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset)); }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) { val += __shfl_down_sync(0xffffffff, val, offset); }
    return val;
}

// CUDA kernel for computing the softmax function (warp-optimized version).
// Each block processes one row of the input matrix.
// Warp-level primitives are used for fast reductions.
__global__ void softmax_forward_kernel_v3(float *__restrict__ output, const float *__restrict__ input, int num_rows,
                                          int num_cols) {
    // ------------------------------------------------------------------
    // Block / thread indexing
    // ------------------------------------------------------------------
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row_idx >= num_rows) {
        return;
    }

    int lane_id = tid & 31; // thread index within warp
    int warp_id = tid >> 5; // warp index within block
    int warps_per_block = block_size >> 5;

    // Pointer to the current row
    const float *input_row = input + row_idx * num_cols;
    float *output_row = output + row_idx * num_cols;

    // ------------------------------------------------------------------
    // Shared memory layout:
    // [0 ... warps_per_block - 1]          -> warp max values
    // [warps_per_block ... 2*warps_per_block - 1] -> warp sum values
    // ------------------------------------------------------------------
    extern __shared__ float shared[];
    float *warp_max = shared;
    float *warp_sum = shared + warps_per_block;

    // ------------------------------------------------------------------
    // Step 1: compute maximum value of the row (numerical stability)
    // Thread coarsening + warp-level reduction
    // ------------------------------------------------------------------
    float local_max = -CUDART_INF_F;
    for (int col = tid; col < num_cols; col += block_size) { local_max = fmaxf(local_max, input_row[col]); }

    // Warp-level max reduction
    local_max = warpReduceMax(local_max);

    // Write warp result to shared memory
    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
    }
    __syncthreads();

    // Block-level reduction across warps
    if (tid == 0) {
        float max_val = warp_max[0];
        for (int i = 1; i < warps_per_block; ++i) { max_val = fmaxf(max_val, warp_max[i]); }
        warp_max[0] = max_val;
    }
    __syncthreads();

    float max_value = warp_max[0];

    // ------------------------------------------------------------------
    // Step 2: compute exponentials and their sum
    // Thread coarsening + warp-level reduction
    // ------------------------------------------------------------------
    float local_sum = 0.0f;
    for (int col = tid; col < num_cols; col += block_size) {
        float exp_val = expf(input_row[col] - max_value);
        output_row[col] = exp_val;
        local_sum += exp_val;
    }

    // Warp-level sum reduction
    local_sum = warpReduceSum(local_sum);

    // Write warp sum to shared memory
    if (lane_id == 0) {
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    // Block-level reduction across warps
    if (tid == 0) {
        float sum_val = warp_sum[0];
        for (int i = 1; i < warps_per_block; ++i) { sum_val += warp_sum[i]; }
        warp_sum[0] = sum_val;
    }
    __syncthreads();

    float sum_exp = warp_sum[0];

    // ------------------------------------------------------------------
    // Step 3: normalize to obtain probabilities
    // ------------------------------------------------------------------
    float inv_sum = 1.0f / sum_exp;
    for (int col = tid; col < num_cols; col += block_size) { output_row[col] *= inv_sum; }
}

int main() {
    // ------------------------------------------------------------
    // Problem size
    // ------------------------------------------------------------
    constexpr int N = 4096;
    constexpr int C = 4096;
    constexpr size_t num_elements = size_t(N) * C;
    constexpr size_t bytes = num_elements * sizeof(float);

    // Steady-state parameters
    constexpr int WARMUP_ITERS = 50;
    constexpr int MEASURE_ITERS = 500;

    std::cout << "N=" << N << ", C=" << C << std::endl;
    std::cout << "Warmup iters=" << WARMUP_ITERS << ", Measure iters=" << MEASURE_ITERS << std::endl;

    // ------------------------------------------------------------
    // Host memory
    // ------------------------------------------------------------
    float *h_inp = (float *)malloc(bytes);
    for (int i = 0; i < num_elements; ++i) { h_inp[i] = static_cast<float>(i % C); }

    // ------------------------------------------------------------
    // Device memory
    // ------------------------------------------------------------
    float *d_inp = nullptr;
    float *d_out = nullptr;
    cudaMalloc(&d_inp, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_inp, h_inp, bytes, cudaMemcpyHostToDevice);

    // ------------------------------------------------------------
    // Launch config
    // ------------------------------------------------------------
    constexpr int block_size = 1024;
    dim3 block(block_size);
    dim3 grid(N);                          // one block per row
    size_t smem = block.x * sizeof(float); // shared memory for reductions

    // ------------------------------------------------------------
    // Warm-up phase (NOT timed)
    // ------------------------------------------------------------
    for (int i = 0; i < WARMUP_ITERS; ++i) { softmax_forward_kernel_v3<<<grid, block, smem>>>(d_out, d_inp, N, C); }
    cudaDeviceSynchronize();

    // ------------------------------------------------------------
    // Steady-state measurement
    // ------------------------------------------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < MEASURE_ITERS; ++i) { softmax_forward_kernel_v3<<<grid, block, smem>>>(d_out, d_inp, N, C); }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_time_ms = 0.0f;
    cudaEventElapsedTime(&total_time_ms, start, stop);

    float avg_kernel_time_us = (total_time_ms * 1000.0f) / MEASURE_ITERS;

    std::cout << "Average steady-state kernel time: " << avg_kernel_time_us << " us" << std::endl;

    // ------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_inp);
    cudaFree(d_out);
    free(h_inp);

    return 0;
}