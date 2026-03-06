#include "assert.h"
#include "cuda_runtime.h"

#include "cmath"
#include <fstream>
#include <iostream>
#include <vector>

#include "util.h"

// ------------------------------------------------------------
// CUDA Error Checking Macro
// ------------------------------------------------------------
#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            printf("CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// ------------------------------------------------------------
// Debug Switch
// ------------------------------------------------------------
#ifdef DEBUG
#define DEBUG_BLOCK(expr)                                                                                              \
    do { expr } while (0)
#else
#define DEBUG_BLOCK(...)
#endif

// ============================================================
// CUDA Kernels
// ============================================================

/*
 * Kernel: naive_nrow_gemm
 *
 * Computes:
 *     C = alpha * (A * B^T) + beta * C
 *
 * Matrix shapes:
 *     A : [M x K]
 *     B : [N x K]
 *     C : [M x N]
 *
 * Each thread processes mBlock rows of A.
 */
__global__ void naive_nrow_gemm(float *A, float *B, float *C, float alpha, float beta, int M, int N, int K,
                                int mBlock) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x; // threadIdx
    int row_start = tid * mBlock;

    for (int i = row_start; i < row_start + mBlock; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;

            for (int k = 0; k < K; k++) { sum += A[i * K + k] * B[j * K + k]; }

            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

/*
 * Kernel: naive_pv
 *
 * Computes matrix multiplication:
 *     O = P * V
 *
 * Matrix shapes:
 *     P : [M x M]
 *     V : [M x N]
 *     O : [M x N]
 *
 * Each thread processes mBlock rows.
 */
__global__ void naive_pv(float *P, float *V, float *O, int M, int N, int mBlock) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int row_start = tid * mBlock;
    int K = M;

    for (int i = row_start; i < row_start + mBlock; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.f;

            for (int k = 0; k < K; k++) { sum += P[i * K + k] * V[k * N + j]; }

            O[i * N + j] = sum;
        }
    }
}

/*
 * Kernel: row_softmax
 *
 * Computes softmax across each row of a matrix.
 *
 * input  : [n x n]
 * output : [n x n]
 *
 * Each thread handles one row.
 */
__global__ void row_softmax(float *input, float *output, int n) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;

    float max_val = -INFINITY;
    float sum = 0.f;

    for (int i = 0; i < n; i++) {
        float v = input[row * n + i];
        if (v > max_val) {
            max_val = v;
        }
    }

    // Compute exponentials
    for (int i = 0; i < n; i++) {
        float e = expf(input[row * n + i] - max_val);
        output[row * n + i] = e;
        sum += e;
    }

    // Normalize
    for (int i = 0; i < n; i++) { output[row * n + i] /= sum; }
}

// ============================================================
// Self-Attention CUDA Implementation
// ============================================================

/*
 * Compute self-attention:
 *
 *      O = softmax(QK^T / sqrt(d)) * V
 *
 * Q : [M x N]
 * K : [M x N]
 * V : [M x N]
 * O : [M x N]
 */
void self_attention_cuda(float *Q, float *K, float *V, float *O, int M, int N) {
    int mBlock = 2;

    assert(M % mBlock == 0);

    float scale = 1.f / sqrtf((float)N);
    float *S = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&S, sizeof(float) * M * N));

    // --------------------------------------------------------
    // Step 1: Compute scaled attention scores
    //         S = QK^T / sqrt(d)
    // --------------------------------------------------------
    dim3 qk_threads(M / mBlock, 1, 1);

    naive_nrow_gemm<<<1, qk_threads>>>(Q, K, S, scale, 0, M, M, N, mBlock);

    cudaDeviceSynchronize();

    DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError()); printf("QK^T result\n"); print_device_matrix(S, M, M););

    // --------------------------------------------------------
    // Step 2: Row-wise softmax
    // --------------------------------------------------------
    dim3 sm_threads(M, 1, 1);

    row_softmax<<<1, sm_threads>>>(S, S, M);

    cudaDeviceSynchronize();

    DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError()); printf("Softmax(QK)\n"); print_device_matrix(S, M, M););

    // --------------------------------------------------------
    // Step 3: Compute output
    //         O = P * V
    // --------------------------------------------------------
    dim3 pv_threads(M / mBlock, 1, 1);

    naive_pv<<<1, pv_threads>>>(S, V, O, M, N, mBlock);

    cudaDeviceSynchronize();

    DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError()); printf("Output O\n"); print_device_matrix(O, M, N););

    cudaFree(S);
}

// ============================================================
// Binary File Utilities
// ============================================================

/*
 * Read floating-point data from a binary file.
 */
bool read_bin(const char *filename, float *data, size_t num_elements) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        printf("Failed to open file: %s\n", filename);
        return false;
    }

    file.read(reinterpret_cast<char *>(data), num_elements * sizeof(float));

    if (!file) {
        printf("Failed to read file: %s\n", filename);
        file.close();

        return false;
    }

    file.close();

    printf("Loadded file: %s (%zu elsements)\n", filename, num_elements);

    return true;
}

/*
 * Write floating-point data to a binary file.
 */
bool write_bin(const char *filename, const float *data, size_t num_elements) {
    std::ofstream file(filename, std::ios::binary);

    if (!file) {
        printf("Failed to create file: %s\n", filename);
        return false;
    }

    file.write(reinterpret_cast<const char *>(data), num_elements * sizeof(float));
    file.close();

    printf("Saved file: %s (%zu elements)\n", filename, num_elements);

    return true;
}

// ============================================================
// Self-Attention with File I/O
// ============================================================

void self_attention_with_io(int M, int N) {
    size_t num_elements = M * N;

    // --------------------------------------------------------
    // Allocate host memory
    // --------------------------------------------------------
    float *h_Q = new float[num_elements];
    float *h_K = new float[num_elements];
    float *h_V = new float[num_elements];
    float *h_O = new float[num_elements];

    // --------------------------------------------------------
    // Load input matrices
    // --------------------------------------------------------
    read_bin("./data/Q.bin", h_Q, num_elements);
    read_bin("./data/K.bin", h_K, num_elements);
    read_bin("./data/V.bin", h_V, num_elements);

    // --------------------------------------------------------
    // Allocate device memory
    // --------------------------------------------------------
    float *d_Q, *d_K, *d_V, *d_O;

    CUDA_CHECK(cudaMalloc(&d_Q, num_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, num_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, num_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, num_elements * sizeof(float)));

    // --------------------------------------------------------
    // Copy inputs to device
    // --------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, num_elements * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_K, h_K, num_elements * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_V, h_V, num_elements * sizeof(float), cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    // Run self-attention
    // --------------------------------------------------------
    self_attention_cuda(d_Q, d_K, d_V, d_O, M, N);

    // --------------------------------------------------------
    // Copy result back to host
    // --------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_O, d_O, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    // --------------------------------------------------------
    // Save output
    // --------------------------------------------------------
    write_bin("./output/O_cuda.bin", h_O, num_elements);

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    printf("Self-attention computation finished\n");
}

// ============================================================
// Program Entry
// ============================================================

int main() {
    const int M = 64;
    const int N = 128;

    printf("Running self-attention (M=%d, N=%d)\n", M, N);

    self_attention_with_io(M, N);

    return 0;
}