#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

// CUDA kernel for computing the softmax function (optimized version).
// Each block processes one row of the input matrix.
// Threads within the block cooperate to compute softmax.
__global__ void softmax_forward_kernel_v2(float *__restrict__ output, const float *__restrict__ input, int num_rows,
                                          int num_cols) {
    // Row index handled by this block
    int row_idx = blockIdx.x;

    // Thread index within the block
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Guard against out-of-bounds blocks
    if (row_idx >= num_rows) {
        return;
    }

    // Pointer to the current row
    const float *input_row = input + row_idx * num_cols;
    float *output_row = output + row_idx * num_cols;

    // Shared memory for block-level reductions
    extern __shared__ float shared[];

    // ------------------------------------------------------------------
    // Step 1: compute the maximum value of the row (numerical stability)
    // Each thread processes multiple columns (thread coarsening)
    // ------------------------------------------------------------------
    float local_max = -CUDART_INF_F;
    for (int col = tid; col < num_cols; col += block_size) { local_max = fmaxf(local_max, input_row[col]); }

    // Write partial maxima to shared memory
    shared[tid] = local_max;
    __syncthreads();

    // Block-level reduction to find the maximum value
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }

        __syncthreads();
    }

    // Broadcast the maximum value to all threads
    float max_value = shared[0];

    // ------------------------------------------------------------------
    // Step 2: compute exponentials and their sum
    // Each thread again processes multiple columns
    // ------------------------------------------------------------------
    float local_sum = 0.0f;
    for (int col = tid; col < num_cols; col += block_size) {
        float exp_val = expf(input_row[col] - max_value);
        output_row[col] = exp_val;
        local_sum += exp_val;
    }

    // Write partial sums to shared memory
    shared[tid] = local_sum;
    __syncthreads();

    // Block-level reduction to compute the sum of exponentials
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // Broadcast the sum to all threads
    float sum_exp = shared[0];

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
    for (int i = 0; i < WARMUP_ITERS; ++i) { softmax_forward_kernel_v2<<<grid, block, smem>>>(d_out, d_inp, N, C); }
    cudaDeviceSynchronize();

    // ------------------------------------------------------------
    // Steady-state measurement
    // ------------------------------------------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < MEASURE_ITERS; ++i) { softmax_forward_kernel_v2<<<grid, block, smem>>>(d_out, d_inp, N, C); }

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