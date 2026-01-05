#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

// CUDA kernel for computing the softmax function.
// Each thread processes one row of the input matrix.
__global__ void softmax_forward_kernel_v1(float *__restrict__ output,      // [N, C] output tensor
                                          const float *__restrict__ input, // [N, C] input tensor
                                          int num_rows,                    // N: number of rows
                                          int num_cols                     // C: number of columns per row
) {
    // Global thread index corresponding to the row index
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against out-of-bounds threads
    if (row_idx >= num_rows) {
        return;
    }

    // Pointers to the current row
    const float *input_row = input + row_idx * num_cols;
    float *output_row = output + row_idx * num_cols;

    // Step 1: find the maximum value in the row (for numerical stability)
    float max_value = -CUDART_INF_F;
    for (int col = 0; col < num_cols; ++col) { max_value = fmaxf(max_value, input_row[col]); }

    // Step 2: compute exponentials and their sum
    float sum_exp = 0.0f;
    for (int col = 0; col < num_cols; ++col) {
        float exp_val = expf(input_row[col] - max_value);
        output_row[col] = exp_val;
        sum_exp += exp_val;
    }

    // Step 3: normalize to obtain probabilities
    float inv_sum = 1.0f / sum_exp;
    for (int col = 0; col < num_cols; ++col) { output_row[col] *= inv_sum; }
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
    constexpr int block_size = 128;
    dim3 block(block_size);
    dim3 grid((N + block.x - 1) / block.x);

    // ------------------------------------------------------------
    // Warm-up phase (NOT timed)
    // ------------------------------------------------------------
    for (int i = 0; i < WARMUP_ITERS; ++i) { softmax_forward_kernel_v1<<<grid, block>>>(d_out, d_inp, N, C); }
    cudaDeviceSynchronize();

    // ------------------------------------------------------------
    // Steady-state measurement
    // ------------------------------------------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < MEASURE_ITERS; ++i) { softmax_forward_kernel_v1<<<grid, block>>>(d_out, d_inp, N, C); }

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