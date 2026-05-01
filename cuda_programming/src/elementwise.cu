#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32

// 索引计算宏定义
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BF16_2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])


__global__ void elementwise_add_f32_kernel(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void elementwise_add_f32x4_kernel(const float4 *A, const float4 *B, float4 *C, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        float4 reg_a = FLOAT4(A[idx]);
        float4 reg_b = FLOAT4(B[idx]);
        float4 reg_c;

        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FLOAT4(C[idx]) = reg_c;
    }
}

__global__ void elementwise_add_f16_kernel(const half *A, const half *B, half *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = __hadd(A[idx], B[idx]);
    }
}

__global__ void elementwise_add_f16x2_kernel(const half2 *A, const half2 *B, half2 *C, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        half2 reg_a = HALF2(A[idx]);
        half2 reg_b = HALF2(B[idx]);
        half2 reg_c;

        reg_c.x = __hadd(reg_a.x, reg_b.x);
        reg_c.y = __hadd(reg_a.y, reg_b.y);
        HALF2(C[idx]) = reg_c;
    }
}

__global__ void elementwise_add_f16x8_kernel(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        half2 reg_a_0 = HALF2(A[idx + 0]);
        half2 reg_a_1 = HALF2(A[idx + 2]);
        half2 reg_a_2 = HALF2(A[idx + 4]);
        half2 reg_a_3 = HALF2(A[idx + 6]);
        half2 reg_b_0 = HALF2(B[idx + 0]);
        half2 reg_b_1 = HALF2(B[idx + 2]);
        half2 reg_b_2 = HALF2(B[idx + 4]);
        half2 reg_b_3 = HALF2(B[idx + 6]);
        half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;

        reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
        reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
        reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
        reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
        reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
        reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
        reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
        reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);

        if ((idx + 0) < N) HALF2(C[idx + 0]) = reg_c_0;
        if ((idx + 2) < N) HALF2(C[idx + 2]) = reg_c_1;
        if ((idx + 4) < N) HALF2(C[idx + 4]) = reg_c_2;
        if ((idx + 6) < N) HALF2(C[idx + 6]) = reg_c_3;
    }
}

__global__ void elementwise_add_f16x8_pack_kernel(const half *A, const half *B, half *C, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half pack_A[8], pack_B[8], pack_C[8];
    
    LDST128BITS(pack_A[0]) = LDST128BITS(A[idx]);
    LDST128BITS(pack_B[0]) = LDST128BITS(B[idx]);

    for (int i = 0; i < 8; i += 2) {
        HALF2(pack_C[i]) = __hadd2(HALF2(pack_A[i]), HALF2(pack_B[i]));
    }
    
    if ((idx + 7) < N) {
        LDST128BITS(C[idx]) = LDST128BITS(pack_C[0]);
    } else {
        for (int i = 0; i < 8; i++) {
            C[idx + i] = __hadd(A[idx + i], B[idx + i]);
        }
    }
}

#define STRINGIFY(str) #str
#define T
